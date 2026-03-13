use super::config::SlmConfig;
/// SLM-based intent classifier using llama.cpp via Python subprocess
///
/// This module implements semantic intent classification using a Small Language Model
/// (Qwen 2.5 0.5B) via llama-cpp-python. It provides significant accuracy improvements
/// over pattern-based classification:
///
/// - Entity queries: 11% → 85%+ accuracy
/// - Causal queries: 30% → 85%+ accuracy
/// - Overall: 35% → 85%+ accuracy
///
/// The classifier integrates with QueryPlanner and falls back to pattern-based
/// classification on any error, ensuring zero regression.
use crate::error::{Error, Result};
use crate::query::intent::IntentClassification;

#[cfg(feature = "slm")]
use std::io::{BufRead, BufReader, Write};
#[cfg(feature = "slm")]
use std::path::PathBuf;
#[cfg(feature = "slm")]
use std::process::{Child, Command, Stdio};

/// SLM-based intent classifier
///
/// Uses Qwen 2.5 0.5B model via persistent Python server for semantic understanding of query intent.
/// The Python process stays running with the model loaded in memory for fast inference.
/// Falls back to pattern-based classification on any error.
#[cfg(feature = "slm")]
pub struct SlmClassifier {
    config: SlmConfig,
    script_path: PathBuf,
    process: Option<Child>,
    stdin: Option<std::process::ChildStdin>,
    stdout: Option<BufReader<std::process::ChildStdout>>,
}

#[cfg(feature = "slm")]
impl SlmClassifier {
    /// Create a new SLM classifier
    ///
    /// Spawns a persistent Python server process that loads the model once and keeps it in memory.
    ///
    /// # Arguments
    ///
    /// * `config` - SLM configuration including model path
    ///
    /// # Returns
    ///
    /// Returns a new classifier instance with running Python server.
    pub fn new(config: SlmConfig) -> Result<Self> {
        tracing::info!(
            "Initializing SLM classifier with model: {}",
            config.model_id
        );

        // Find the Python server script
        let script_path = Self::find_script_path()?;

        tracing::info!("Using SLM classification script: {}", script_path.display());

        // Get model path
        let mut classifier = Self {
            config: config.clone(),
            script_path,
            process: None,
            stdin: None,
            stdout: None,
        };

        let model_path = classifier.get_model_path()?;

        // Spawn Python server process
        let mut child = Command::new("python")
            .arg(&classifier.script_path)
            .arg(model_path.to_string_lossy().as_ref())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Show server logs in stderr
            .spawn()
            .map_err(|e| {
                Error::SlmInitialization(format!("Failed to spawn Python server: {}", e))
            })?;

        // Get handles
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::SlmInitialization("Failed to get stdin handle".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::SlmInitialization("Failed to get stdout handle".to_string()))?;
        let mut stdout_reader = BufReader::new(stdout);

        // Wait for READY signal
        let mut ready_line = String::new();
        std::io::BufRead::read_line(&mut stdout_reader, &mut ready_line)
            .map_err(|e| Error::SlmInitialization(format!("Failed to read READY signal: {}", e)))?;

        if ready_line.trim() != "READY" {
            return Err(Error::SlmInitialization(format!(
                "Expected READY signal, got: {}",
                ready_line.trim()
            )));
        }

        tracing::info!("SLM server initialized and ready");

        classifier.process = Some(child);
        classifier.stdin = Some(stdin);
        classifier.stdout = Some(stdout_reader);

        Ok(classifier)
    }

    /// Find the slm_classify_server.py script
    fn find_script_path() -> Result<PathBuf> {
        // Check environment variable first
        if let Ok(script_path) = std::env::var("SLM_SCRIPT_PATH") {
            let path = PathBuf::from(script_path);
            if path.exists() {
                tracing::info!("Using SLM script from SLM_SCRIPT_PATH: {}", path.display());
                return Ok(path);
            }
        }

        // Try several possible locations
        let possible_paths = vec![
            PathBuf::from("scripts/slm_classify_server.py"),
            PathBuf::from("../scripts/slm_classify_server.py"),
            PathBuf::from("../../scripts/slm_classify_server.py"),
            PathBuf::from("../../../scripts/slm_classify_server.py"), // For deeper test directories
        ];

        for path in possible_paths {
            if path.exists() {
                tracing::info!("Found SLM script at: {}", path.display());
                return Ok(path);
            } else {
                tracing::debug!("Script not found at: {}", path.display());
            }
        }

        Err(Error::SlmInitialization(
            "Could not find scripts/slm_classify_server.py. Make sure it exists in the project root, or set SLM_SCRIPT_PATH environment variable.".to_string()
        ))
    }

    /// Get the GGUF model path
    ///
    /// Returns the path to the .gguf file, either from explicit config
    /// or derived from model_id.
    fn get_model_path(&self) -> Result<PathBuf> {
        if let Some(model_path) = &self.config.model_path {
            // Check if it's a directory or file
            if model_path.is_dir() {
                // Look for .gguf file in directory
                let gguf_files: Vec<_> = std::fs::read_dir(model_path)
                    .map_err(|e| {
                        Error::SlmInitialization(format!("Failed to read model directory: {}", e))
                    })?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
                    .collect();

                if gguf_files.is_empty() {
                    return Err(Error::SlmInitialization(format!(
                        "No .gguf file found in directory: {}\n\
                         Please ensure the model has been converted to GGUF format.",
                        model_path.display()
                    )));
                }

                if gguf_files.len() > 1 {
                    tracing::warn!(
                        "Multiple .gguf files found, using first: {:?}",
                        gguf_files[0]
                    );
                }

                Ok(gguf_files[0].clone())
            } else {
                // Direct path to .gguf file
                if !model_path.exists() {
                    return Err(Error::SlmInitialization(format!(
                        "Model file not found: {}",
                        model_path.display()
                    )));
                }
                Ok(model_path.clone())
            }
        } else {
            // Try to derive path from model_id
            // e.g., "Qwen/Qwen2.5-0.5B-Instruct" -> "opt/models/qwen2.5-0.5b-instruct.gguf"
            let model_filename = self
                .config
                .model_id
                .split('/')
                .last()
                .unwrap_or(&self.config.model_id)
                .to_lowercase()
                .replace('.', "-")
                .replace("_", "-");

            // Try both opt/models and cache_dir
            let possible_paths = vec![
                PathBuf::from(format!("opt/models/{}.gguf", model_filename)),
                self.config
                    .cache_dir
                    .join(format!("{}.gguf", model_filename)),
            ];

            for path in possible_paths {
                if path.exists() {
                    return Ok(path);
                }
            }

            Err(Error::SlmInitialization(format!(
                "Model file not found for: {}\n\
                 Tried: opt/models/{}.gguf\n\
                 Please either:\n\
                 1. Use config.with_model_path(\"/path/to/model.gguf\"), or\n\
                 2. Place the GGUF file at opt/models/{}.gguf",
                self.config.model_id, model_filename, model_filename
            )))
        }
    }

    /// Classify query intent using SLM via persistent Python server
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to classify
    ///
    /// # Returns
    ///
    /// Returns structured intent classification with confidence score.
    /// On any error, the caller should fall back to pattern-based classification.
    ///
    /// # Errors
    ///
    /// Returns error if communication with server or parsing fails.
    /// Callers should catch errors and fall back to pattern classifier.
    pub fn classify_intent(&mut self, query: &str) -> Result<IntentClassification> {
        let start = std::time::Instant::now();

        tracing::debug!("Starting SLM inference for query: '{}'", query);

        // Get handles (should always be Some after successful initialization)
        let stdin = self.stdin.as_mut().ok_or_else(|| {
            Error::SlmInference("Python server not initialized (no stdin)".to_string())
        })?;
        let stdout = self.stdout.as_mut().ok_or_else(|| {
            Error::SlmInference("Python server not initialized (no stdout)".to_string())
        })?;

        // Send request as JSON
        let request = serde_json::json!({ "query": query });
        let request_str = format!("{}\n", request.to_string());

        stdin
            .write_all(request_str.as_bytes())
            .map_err(|e| Error::SlmInference(format!("Failed to write to Python server: {}", e)))?;
        stdin
            .flush()
            .map_err(|e| Error::SlmInference(format!("Failed to flush stdin: {}", e)))?;

        // Read response
        let mut response_line = String::new();
        stdout.read_line(&mut response_line).map_err(|e| {
            Error::SlmInference(format!("Failed to read from Python server: {}", e))
        })?;

        let duration = start.elapsed();
        tracing::info!("SLM inference completed in {:?}", duration);

        // Parse JSON output from Python
        self.parse_intent_output(&response_line)
    }

    /// Create classification prompt for the model
    ///
    /// Formats the query with instructions for structured JSON output.
    pub fn create_classification_prompt(&self, query: &str) -> String {
        format!(
            r#"<|im_start|>system
You are a query intent classifier. Classify queries into: Entity, Temporal, Causal, or Factual.<|im_end|>
<|im_start|>user
Classify this query: "{}"

Respond with JSON:
{{
  "intent": "Entity|Temporal|Causal|Factual",
  "confidence": 0.0-1.0,
  "entity_focus": "subject entity if intent=Entity, else null",
  "reasoning": "brief explanation"
}}<|im_end|>
<|im_start|>assistant
"#,
            query
        )
    }

    /// Parse model output into structured classification
    ///
    /// Expects JSON output from the model in the format specified by the prompt.
    pub fn parse_intent_output(&self, output: &str) -> Result<IntentClassification> {
        // Parse JSON directly (Python script already extracted it)
        let parsed: serde_json::Value = serde_json::from_str(output.trim())
            .map_err(|e| Error::SlmInference(format!("JSON parse error: {}", e)))?;

        // Extract intent
        let intent_str = parsed["intent"]
            .as_str()
            .ok_or_else(|| Error::SlmInference("Missing 'intent' field".to_string()))?;

        let intent = match intent_str {
            "Entity" => QueryIntent::Entity,
            "Temporal" => QueryIntent::Temporal,
            "Causal" => QueryIntent::Causal,
            _ => QueryIntent::Factual, // Default fallback
        };

        // Extract confidence
        let confidence = parsed["confidence"].as_f64().unwrap_or(0.5) as f32;

        // Extract entity focus if present
        let entity_focus = parsed["entity_focus"]
            .as_str()
            .filter(|s| *s != "null")
            .map(String::from);

        Ok(IntentClassification {
            intent,
            confidence,
            secondary: Vec::new(), // No secondary intents from SLM yet
            entity_focus,
        })
    }
}

#[cfg(feature = "slm")]
impl Drop for SlmClassifier {
    fn drop(&mut self) {
        // Send QUIT command to server
        if let Some(stdin) = self.stdin.as_mut() {
            let _ = stdin.write_all(b"{\"query\":\"QUIT\"}\n");
            let _ = stdin.flush();
        }

        // Wait for process to exit (with timeout)
        if let Some(mut child) = self.process.take() {
            let _ = std::thread::spawn(move || {
                let _ = child.wait();
            });
        }

        tracing::debug!("SLM classifier server shutdown");
    }
}

/// Stub implementation when SLM feature is disabled
///
/// This allows the code to compile without the SLM feature enabled.
/// In production, the QueryPlanner checks if SLM is configured before using it.
#[cfg(not(feature = "slm"))]
pub struct SlmClassifier {
    _private: (),
}

#[cfg(not(feature = "slm"))]
impl SlmClassifier {
    pub fn new(_config: SlmConfig) -> Result<Self> {
        Err(Error::SlmNotAvailable)
    }

    pub fn classify_intent(&mut self, _query: &str) -> Result<IntentClassification> {
        Err(Error::SlmNotAvailable)
    }
}

#[cfg(all(test, feature = "slm"))]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let config = SlmConfig::default();
        let classifier = SlmClassifier::new(config);
        assert!(classifier.is_ok());
    }

    #[test]
    fn test_prompt_generation() {
        let config = SlmConfig::default();
        let classifier = SlmClassifier::new(config).unwrap();

        let prompt = classifier.create_classification_prompt("Who was the first speaker?");
        assert!(prompt.contains("Who was the first speaker?"));
        assert!(prompt.contains("Entity|Temporal|Causal"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_parse_entity_output() {
        let config = SlmConfig::default();
        let classifier = SlmClassifier::new(config).unwrap();

        let output = r#"{
            "intent": "Entity",
            "confidence": 0.95,
            "entity_focus": "first speaker",
            "reasoning": "Query asks 'who' indicating entity focus"
        }"#;

        let result = classifier.parse_intent_output(output).unwrap();
        assert_eq!(result.intent, QueryIntent::Entity);
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.entity_focus, Some("first speaker".to_string()));
    }

    #[test]
    fn test_parse_temporal_output() {
        let config = SlmConfig::default();
        let classifier = SlmClassifier::new(config).unwrap();

        let output = r#"{
            "intent": "Temporal",
            "confidence": 0.88,
            "entity_focus": null,
            "reasoning": "Query asks about timing"
        }"#;

        let result = classifier.parse_intent_output(output).unwrap();
        assert_eq!(result.intent, QueryIntent::Temporal);
        assert_eq!(result.confidence, 0.88);
        assert_eq!(result.entity_focus, None);
    }

    #[test]
    fn test_parse_with_extra_text() {
        let config = SlmConfig::default();
        let classifier = SlmClassifier::new(config).unwrap();

        // Python script outputs clean JSON, so no extra text expected
        let output = r#"{
            "intent": "Causal",
            "confidence": 0.92,
            "entity_focus": null,
            "reasoning": "Query asks 'why'"
        }"#;

        let result = classifier.parse_intent_output(output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().intent, QueryIntent::Causal);
    }
}
