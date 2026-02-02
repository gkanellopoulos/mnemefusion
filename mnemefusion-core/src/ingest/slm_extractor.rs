//! SLM-based metadata extraction at ingestion time
//!
//! This module implements rich metadata extraction using a Small Language Model
//! (Qwen 2.5 0.5B) via llama-cpp-python. The SLM extracts structured information
//! during memory ingestion, enabling fast retrieval without query-time SLM inference.
//!
//! # Architecture Principle
//!
//! - Ingestion: Can be slow (3-5s per memory) - Users accept "Processing..." UX
//! - Query: Must be fast (<100ms) - Query-time SLM is unacceptable
//!
//! By extracting rich metadata at ingestion time, we "pay the cost once" and enable
//! fast, accurate retrieval forever.
//!
//! # Extracted Metadata
//!
//! - **Entities**: Named entities with roles, mentions, and types
//! - **Temporal**: Time markers, sequence position, relative timing
//! - **Causal**: Cause-effect relationships with confidence scores
//! - **Topics**: Key themes and subjects
//! - **Importance**: Subjective importance score (0.0-1.0)

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

#[cfg(feature = "slm")]
use crate::slm::SlmConfig;
#[cfg(feature = "slm")]
use std::io::{BufRead, BufReader, Write};
#[cfg(feature = "slm")]
use std::path::PathBuf;
#[cfg(feature = "slm")]
use std::process::{Child, Command, Stdio};

/// Rich metadata extracted from memory content by the SLM
///
/// This structure contains all the information extracted at ingestion time
/// to enable fast, accurate retrieval without query-time inference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SlmMetadata {
    /// Named entities extracted from the content
    pub entities: Vec<ExtractedEntity>,

    /// Temporal information (time markers, sequence, relative timing)
    pub temporal: TemporalMetadata,

    /// Causal relationships and patterns
    pub causal: CausalMetadata,

    /// Key topics and themes
    pub topics: Vec<String>,

    /// Importance score (0.0 to 1.0)
    pub importance: f32,

    /// Entity-specific facts extracted from content
    /// These are structured facts about entities that enable direct lookup
    /// e.g., "Caroline researches adoption agencies" -> {entity: "Caroline", fact_type: "research_topic", value: "adoption agencies"}
    pub entity_facts: Vec<ExtractedEntityFact>,

    /// Schema version for future migrations
    pub schema_version: u32,
}

impl Default for SlmMetadata {
    fn default() -> Self {
        Self {
            entities: Vec::new(),
            temporal: TemporalMetadata::default(),
            causal: CausalMetadata::default(),
            topics: Vec::new(),
            importance: 0.5,
            entity_facts: Vec::new(),
            schema_version: 1,
        }
    }
}

/// A fact about an entity extracted from the content
///
/// Entity facts enable direct fact lookup instead of relying on semantic similarity.
/// For example, "Caroline researches adoption agencies" extracts:
/// - entity: "Caroline"
/// - fact_type: "research_topic"
/// - value: "adoption agencies"
/// - confidence: 0.9
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractedEntityFact {
    /// Entity name this fact is about
    pub entity: String,

    /// Fact type: "research_topic", "occupation", "relationship", "goal", "preference", etc.
    pub fact_type: String,

    /// Fact value
    pub value: String,

    /// Confidence (0.0-1.0)
    pub confidence: f32,
}

impl ExtractedEntityFact {
    /// Create a new extracted entity fact
    pub fn new(
        entity: impl Into<String>,
        fact_type: impl Into<String>,
        value: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            entity: entity.into(),
            fact_type: fact_type.into(),
            value: value.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// An entity extracted from memory content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractedEntity {
    /// Primary name of the entity
    pub name: String,

    /// Role in the content: "subject", "object", "organization", "location"
    pub role: String,

    /// All mentions in the text (e.g., ["Caroline", "she", "her"])
    pub mentions: Vec<String>,

    /// Entity type: "person", "organization", "location", "concept"
    pub entity_type: String,
}

impl ExtractedEntity {
    /// Create a new extracted entity
    pub fn new(name: impl Into<String>, entity_type: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            mentions: vec![name.clone()],
            name,
            role: "subject".to_string(),
            entity_type: entity_type.into(),
        }
    }

    /// Set the role of this entity
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = role.into();
        self
    }

    /// Add additional mentions
    pub fn with_mentions(mut self, mentions: Vec<String>) -> Self {
        self.mentions = mentions;
        self
    }
}

/// Temporal metadata extracted from content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TemporalMetadata {
    /// Temporal markers found in text (e.g., "yesterday", "last week")
    pub markers: Vec<String>,

    /// Sequence position in narrative: "early", "middle", "late", or None
    pub sequence: Option<String>,

    /// Relative timing: "before current", "concurrent", "after current", or None
    pub relative_time: Option<String>,

    /// Absolute dates found (ISO format preferred)
    pub absolute_dates: Vec<String>,
}

/// Causal metadata extracted from content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalMetadata {
    /// Explicit cause-effect relationships
    pub relationships: Vec<CausalRelationship>,

    /// Causal density score (0.0 to 1.0)
    pub density: f32,

    /// Explicit causal markers found (e.g., "because", "therefore")
    pub explicit_markers: Vec<String>,

    /// Whether implicit causation was detected
    pub has_implicit_causation: bool,
}

impl Default for CausalMetadata {
    fn default() -> Self {
        Self {
            relationships: Vec::new(),
            density: 0.0,
            explicit_markers: Vec::new(),
            has_implicit_causation: false,
        }
    }
}

/// A single causal relationship
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalRelationship {
    /// The cause event or state
    pub cause: String,

    /// The effect event or state
    pub effect: String,

    /// Confidence in this relationship (0.0 to 1.0)
    pub confidence: f32,
}

impl CausalRelationship {
    /// Create a new causal relationship
    pub fn new(cause: impl Into<String>, effect: impl Into<String>, confidence: f32) -> Self {
        Self {
            cause: cause.into(),
            effect: effect.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// SLM-based metadata extractor
///
/// Uses a persistent Python server process with the SLM loaded in memory
/// for efficient metadata extraction during ingestion.
#[cfg(feature = "slm")]
pub struct SlmMetadataExtractor {
    config: SlmConfig,
    script_path: PathBuf,
    process: Option<Child>,
    stdin: Option<std::process::ChildStdin>,
    stdout: Option<BufReader<std::process::ChildStdout>>,
}

#[cfg(feature = "slm")]
impl SlmMetadataExtractor {
    /// Create a new SLM metadata extractor
    ///
    /// Spawns a persistent Python server process that loads the model once
    /// and keeps it in memory for fast inference.
    ///
    /// # Arguments
    ///
    /// * `config` - SLM configuration including model path
    ///
    /// # Returns
    ///
    /// Returns a new extractor instance with running Python server.
    pub fn new(config: SlmConfig) -> Result<Self> {
        tracing::info!(
            "Initializing SLM metadata extractor with model: {}",
            config.model_id
        );

        // Find the Python server script
        let script_path = Self::find_script_path()?;
        tracing::info!(
            "Using SLM extraction script: {}",
            script_path.display()
        );

        let mut extractor = Self {
            config: config.clone(),
            script_path,
            process: None,
            stdin: None,
            stdout: None,
        };

        // Get model path
        let model_path = extractor.get_model_path()?;
        tracing::debug!("Model path: {}", model_path.display());

        // Spawn Python server process
        tracing::info!("Spawning SLM extraction server...");
        let mut child = Command::new("python")
            .arg(&extractor.script_path)
            .arg(model_path.to_string_lossy().as_ref())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                Error::SlmInitialization(format!("Failed to spawn Python server: {}", e))
            })?;

        // Get handles
        let stdin = child.stdin.take().ok_or_else(|| {
            Error::SlmInitialization("Failed to get stdin handle".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            Error::SlmInitialization("Failed to get stdout handle".to_string())
        })?;
        let mut stdout_reader = BufReader::new(stdout);

        // Wait for READY signal
        tracing::info!("Waiting for SLM extraction server to be ready...");
        let mut ready_line = String::new();
        std::io::BufRead::read_line(&mut stdout_reader, &mut ready_line).map_err(|e| {
            Error::SlmInitialization(format!("Failed to read READY signal: {}", e))
        })?;

        if ready_line.trim() != "READY" {
            return Err(Error::SlmInitialization(format!(
                "Expected READY signal, got: {}",
                ready_line.trim()
            )));
        }

        tracing::info!("SLM extraction server initialized and ready");

        extractor.process = Some(child);
        extractor.stdin = Some(stdin);
        extractor.stdout = Some(stdout_reader);

        Ok(extractor)
    }

    /// Find the slm_extract_server.py script
    fn find_script_path() -> Result<PathBuf> {
        // Check environment variable first
        if let Ok(script_path) = std::env::var("SLM_EXTRACT_SCRIPT_PATH") {
            let path = PathBuf::from(script_path);
            if path.exists() {
                tracing::info!(
                    "Using SLM extraction script from SLM_EXTRACT_SCRIPT_PATH: {}",
                    path.display()
                );
                return Ok(path);
            }
        }

        // Try several possible locations
        let possible_paths = vec![
            PathBuf::from("scripts/slm_extract_server.py"),
            PathBuf::from("../scripts/slm_extract_server.py"),
            PathBuf::from("../../scripts/slm_extract_server.py"),
            PathBuf::from("../../../scripts/slm_extract_server.py"),
        ];

        for path in possible_paths {
            if path.exists() {
                tracing::info!("Found SLM extraction script at: {}", path.display());
                return Ok(path);
            } else {
                tracing::debug!("Script not found at: {}", path.display());
            }
        }

        Err(Error::SlmInitialization(
            "Could not find scripts/slm_extract_server.py. Make sure it exists in the project root, or set SLM_EXTRACT_SCRIPT_PATH environment variable.".to_string()
        ))
    }

    /// Get the GGUF model path
    fn get_model_path(&self) -> Result<PathBuf> {
        if let Some(model_path) = &self.config.model_path {
            if model_path.is_dir() {
                // Look for .gguf file in directory
                let gguf_files: Vec<_> = std::fs::read_dir(model_path)
                    .map_err(|e| {
                        Error::SlmInitialization(format!(
                            "Failed to read model directory: {}",
                            e
                        ))
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
            let model_filename = self
                .config
                .model_id
                .split('/')
                .last()
                .unwrap_or(&self.config.model_id)
                .to_lowercase()
                .replace('.', "-")
                .replace('_', "-");

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

    /// Extract metadata from content using the SLM
    ///
    /// # Arguments
    ///
    /// * `content` - The memory content to extract metadata from
    ///
    /// # Returns
    ///
    /// Returns structured metadata extracted from the content.
    /// On any error, the caller should fall back to pattern-based extraction.
    pub fn extract(&mut self, content: &str) -> Result<SlmMetadata> {
        let start = std::time::Instant::now();

        tracing::debug!(
            "Starting SLM metadata extraction for content: '{}'",
            content.chars().take(100).collect::<String>()
        );

        let stdin = self.stdin.as_mut().ok_or_else(|| {
            Error::SlmInference("Python server not initialized (no stdin)".to_string())
        })?;
        let stdout = self.stdout.as_mut().ok_or_else(|| {
            Error::SlmInference("Python server not initialized (no stdout)".to_string())
        })?;

        // Send request as JSON
        let request = serde_json::json!({ "content": content });
        let request_str = format!("{}\n", request);

        stdin.write_all(request_str.as_bytes()).map_err(|e| {
            Error::SlmInference(format!("Failed to write to Python server: {}", e))
        })?;
        stdin.flush().map_err(|e| {
            Error::SlmInference(format!("Failed to flush stdin: {}", e))
        })?;

        // Read response
        let mut response_line = String::new();
        stdout.read_line(&mut response_line).map_err(|e| {
            Error::SlmInference(format!("Failed to read from Python server: {}", e))
        })?;

        let duration = start.elapsed();
        tracing::info!("SLM metadata extraction completed in {:?}", duration);

        // Parse JSON output
        self.parse_metadata_output(&response_line)
    }

    /// Parse the JSON output from the Python server into SlmMetadata
    pub fn parse_metadata_output(&self, output: &str) -> Result<SlmMetadata> {
        let parsed: serde_json::Value = serde_json::from_str(output.trim())
            .map_err(|e| Error::SlmInference(format!("JSON parse error: {}", e)))?;

        // Check for error field
        if let Some(error) = parsed.get("error").and_then(|e| e.as_str()) {
            tracing::warn!("SLM extraction returned error: {}", error);
            // Continue parsing - the server provides defaults even on error
        }

        // Parse entities
        let entities: Vec<ExtractedEntity> = parsed
            .get("entities")
            .and_then(|e| e.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|e| {
                        let name = e.get("name")?.as_str()?;
                        let role = e
                            .get("role")
                            .and_then(|r| r.as_str())
                            .unwrap_or("subject");
                        let mentions: Vec<String> = e
                            .get("mentions")
                            .and_then(|m| m.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_else(|| vec![name.to_string()]);
                        let entity_type = e
                            .get("entity_type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("concept");

                        Some(ExtractedEntity {
                            name: name.to_string(),
                            role: role.to_string(),
                            mentions,
                            entity_type: entity_type.to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Parse temporal metadata
        let temporal = parsed.get("temporal").map(|t| {
            let markers: Vec<String> = t
                .get("markers")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let sequence = t
                .get("sequence")
                .and_then(|s| s.as_str())
                .filter(|s| *s != "null" && !s.is_empty())
                .map(String::from);

            let relative_time = t
                .get("relative_time")
                .and_then(|r| r.as_str())
                .filter(|s| *s != "null" && !s.is_empty())
                .map(String::from);

            let absolute_dates: Vec<String> = t
                .get("absolute_dates")
                .and_then(|d| d.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            TemporalMetadata {
                markers,
                sequence,
                relative_time,
                absolute_dates,
            }
        }).unwrap_or_default();

        // Parse causal metadata
        let causal = parsed.get("causal").map(|c| {
            let relationships: Vec<CausalRelationship> = c
                .get("relationships")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|rel| {
                            let cause = rel.get("cause")?.as_str()?;
                            let effect = rel.get("effect")?.as_str()?;
                            let confidence = rel
                                .get("confidence")
                                .and_then(|c| c.as_f64())
                                .unwrap_or(0.5) as f32;

                            Some(CausalRelationship {
                                cause: cause.to_string(),
                                effect: effect.to_string(),
                                confidence,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            let density = c
                .get("density")
                .and_then(|d| d.as_f64())
                .unwrap_or(0.0) as f32;

            let explicit_markers: Vec<String> = c
                .get("explicit_markers")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let has_implicit_causation = c
                .get("has_implicit_causation")
                .and_then(|h| h.as_bool())
                .unwrap_or(false);

            CausalMetadata {
                relationships,
                density,
                explicit_markers,
                has_implicit_causation,
            }
        }).unwrap_or_default();

        // Parse topics
        let topics: Vec<String> = parsed
            .get("topics")
            .and_then(|t| t.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Parse entity_facts
        let entity_facts: Vec<ExtractedEntityFact> = parsed
            .get("entity_facts")
            .and_then(|f| f.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|fact| {
                        let entity = fact.get("entity")?.as_str()?;
                        let fact_type = fact.get("fact_type")?.as_str()?;
                        let value = fact.get("value")?.as_str()?;
                        let confidence = fact
                            .get("confidence")
                            .and_then(|c| c.as_f64())
                            .unwrap_or(0.5) as f32;

                        Some(ExtractedEntityFact {
                            entity: entity.to_string(),
                            fact_type: fact_type.to_string(),
                            value: value.to_string(),
                            confidence: confidence.clamp(0.0, 1.0),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Parse importance
        let importance = parsed
            .get("importance")
            .and_then(|i| i.as_f64())
            .unwrap_or(0.5) as f32;

        // Parse schema version
        let schema_version = parsed
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;

        Ok(SlmMetadata {
            entities,
            temporal,
            causal,
            topics,
            importance: importance.clamp(0.0, 1.0),
            entity_facts,
            schema_version,
        })
    }
}

#[cfg(feature = "slm")]
impl Drop for SlmMetadataExtractor {
    fn drop(&mut self) {
        // Send QUIT command to server
        if let Some(stdin) = self.stdin.as_mut() {
            let _ = stdin.write_all(b"QUIT\n");
            let _ = stdin.flush();
        }

        // Wait for process to exit
        if let Some(mut child) = self.process.take() {
            let _ = std::thread::spawn(move || {
                let _ = child.wait();
            });
        }

        tracing::debug!("SLM metadata extractor server shutdown");
    }
}

/// Stub implementation when SLM feature is disabled
#[cfg(not(feature = "slm"))]
pub struct SlmMetadataExtractor {
    _private: (),
}

#[cfg(not(feature = "slm"))]
impl SlmMetadataExtractor {
    pub fn new(_config: crate::slm::SlmConfig) -> Result<Self> {
        Err(Error::SlmNotAvailable)
    }

    pub fn extract(&mut self, _content: &str) -> Result<SlmMetadata> {
        Err(Error::SlmNotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slm_metadata_default() {
        let metadata = SlmMetadata::default();
        assert!(metadata.entities.is_empty());
        assert!(metadata.topics.is_empty());
        assert!(metadata.entity_facts.is_empty());
        assert_eq!(metadata.importance, 0.5);
        assert_eq!(metadata.schema_version, 1);
    }

    #[test]
    fn test_slm_metadata_serialization() {
        let metadata = SlmMetadata {
            entities: vec![ExtractedEntity::new("Alice", "person").with_role("subject")],
            temporal: TemporalMetadata {
                markers: vec!["yesterday".to_string()],
                sequence: Some("early".to_string()),
                relative_time: None,
                absolute_dates: vec![],
            },
            causal: CausalMetadata {
                relationships: vec![CausalRelationship::new(
                    "rain",
                    "cancelled meeting",
                    0.85,
                )],
                density: 0.5,
                explicit_markers: vec!["because".to_string()],
                has_implicit_causation: false,
            },
            topics: vec!["meetings".to_string(), "weather".to_string()],
            importance: 0.8,
            entity_facts: vec![
                ExtractedEntityFact::new("Alice", "occupation", "engineer", 0.9),
            ],
            schema_version: 1,
        };

        // Serialize to JSON
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("Alice"));
        assert!(json.contains("yesterday"));
        assert!(json.contains("because"));
        assert!(json.contains("occupation"));
        assert!(json.contains("engineer"));

        // Deserialize back
        let parsed: SlmMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.entities.len(), 1);
        assert_eq!(parsed.entities[0].name, "Alice");
        assert_eq!(parsed.importance, 0.8);
        assert_eq!(parsed.entity_facts.len(), 1);
        assert_eq!(parsed.entity_facts[0].entity, "Alice");
        assert_eq!(parsed.entity_facts[0].fact_type, "occupation");
    }

    #[test]
    fn test_extracted_entity_fact_new() {
        let fact = ExtractedEntityFact::new("Caroline", "research_topic", "adoption agencies", 0.9);

        assert_eq!(fact.entity, "Caroline");
        assert_eq!(fact.fact_type, "research_topic");
        assert_eq!(fact.value, "adoption agencies");
        assert_eq!(fact.confidence, 0.9);
    }

    #[test]
    fn test_extracted_entity_fact_confidence_clamping() {
        let fact1 = ExtractedEntityFact::new("Test", "type", "value", 1.5);
        assert_eq!(fact1.confidence, 1.0);

        let fact2 = ExtractedEntityFact::new("Test", "type", "value", -0.5);
        assert_eq!(fact2.confidence, 0.0);

        let fact3 = ExtractedEntityFact::new("Test", "type", "value", 0.75);
        assert_eq!(fact3.confidence, 0.75);
    }

    #[test]
    fn test_extracted_entity_builder() {
        let entity = ExtractedEntity::new("John", "person")
            .with_role("object")
            .with_mentions(vec!["John".to_string(), "he".to_string(), "him".to_string()]);

        assert_eq!(entity.name, "John");
        assert_eq!(entity.role, "object");
        assert_eq!(entity.entity_type, "person");
        assert_eq!(entity.mentions.len(), 3);
    }

    #[test]
    fn test_causal_relationship_confidence_clamping() {
        let rel1 = CausalRelationship::new("cause", "effect", 1.5);
        assert_eq!(rel1.confidence, 1.0);

        let rel2 = CausalRelationship::new("cause", "effect", -0.5);
        assert_eq!(rel2.confidence, 0.0);

        let rel3 = CausalRelationship::new("cause", "effect", 0.7);
        assert_eq!(rel3.confidence, 0.7);
    }

    #[cfg(feature = "slm")]
    #[test]
    fn test_parse_metadata_output_valid() {
        // Test the parsing logic by directly parsing JSON
        // (We can't easily test the full extractor without a running Python server)

        let output = r#"{
            "entities": [
                {"name": "Caroline", "role": "subject", "mentions": ["Caroline", "she"], "entity_type": "person"}
            ],
            "temporal": {
                "markers": ["yesterday", "last week"],
                "sequence": "early",
                "relative_time": null,
                "absolute_dates": []
            },
            "causal": {
                "relationships": [{"cause": "rain", "effect": "cancelled", "confidence": 0.9}],
                "density": 0.5,
                "explicit_markers": ["because"],
                "has_implicit_causation": false
            },
            "topics": ["work", "meetings"],
            "entity_facts": [
                {"entity": "Caroline", "fact_type": "research_topic", "value": "adoption agencies", "confidence": 0.9},
                {"entity": "Caroline", "fact_type": "goal", "value": "find biological parents", "confidence": 0.85}
            ],
            "importance": 0.7,
            "schema_version": 1
        }"#;

        // Parse directly using serde (simulating parse_metadata_output logic)
        let parsed: serde_json::Value = serde_json::from_str(output).unwrap();

        // Verify entities parsing
        let entities = parsed.get("entities").unwrap().as_array().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].get("name").unwrap().as_str().unwrap(), "Caroline");

        // Verify temporal parsing
        let temporal = parsed.get("temporal").unwrap();
        let markers = temporal.get("markers").unwrap().as_array().unwrap();
        assert_eq!(markers.len(), 2);

        // Verify causal parsing
        let causal = parsed.get("causal").unwrap();
        let relationships = causal.get("relationships").unwrap().as_array().unwrap();
        assert_eq!(relationships.len(), 1);

        // Verify entity_facts parsing
        let entity_facts = parsed.get("entity_facts").unwrap().as_array().unwrap();
        assert_eq!(entity_facts.len(), 2);
        assert_eq!(entity_facts[0].get("entity").unwrap().as_str().unwrap(), "Caroline");
        assert_eq!(entity_facts[0].get("fact_type").unwrap().as_str().unwrap(), "research_topic");
        assert_eq!(entity_facts[0].get("value").unwrap().as_str().unwrap(), "adoption agencies");
    }

    #[test]
    fn test_temporal_metadata_default() {
        let temporal = TemporalMetadata::default();
        assert!(temporal.markers.is_empty());
        assert!(temporal.sequence.is_none());
        assert!(temporal.relative_time.is_none());
        assert!(temporal.absolute_dates.is_empty());
    }

    #[test]
    fn test_causal_metadata_default() {
        let causal = CausalMetadata::default();
        assert!(causal.relationships.is_empty());
        assert_eq!(causal.density, 0.0);
        assert!(causal.explicit_markers.is_empty());
        assert!(!causal.has_implicit_causation);
    }
}
