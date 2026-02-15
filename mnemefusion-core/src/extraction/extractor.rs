//! LlmEntityExtractor - High-level API for entity extraction

use crate::error::{Error, Result};
use crate::extraction::output::ExtractionResult;
use crate::extraction::prompt::build_fewshot_extraction_prompt;
use crate::inference::InferenceEngine;
use std::path::PathBuf;

/// Model size/capability tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    /// Qwen 3.5-4B: Fast, edge-friendly, good for standard extraction
    /// ~2.5GB model size, recommended for most use cases
    Balanced,

    /// Qwen 3.5-7B: Maximum accuracy, complex multi-entity reasoning
    /// ~4GB model size, use when accuracy is critical
    Quality,
}

impl ModelTier {
    /// Get the expected model filename
    pub fn model_filename(&self) -> &'static str {
        match self {
            ModelTier::Balanced => "Qwen3-4B-Instruct-2507.Q4_K_M.gguf",
            ModelTier::Quality => "Qwen3-8B.Q4_K_M.gguf",
        }
    }

    /// Get default GPU layers for this model tier
    pub fn default_gpu_layers(&self) -> u32 {
        match self {
            ModelTier::Balanced => 99, // All layers on GPU for 4B
            ModelTier::Quality => 99,  // All layers on GPU for 7B
        }
    }

    /// Get recommended max tokens for generation
    pub fn max_tokens(&self) -> u32 {
        match self {
            ModelTier::Balanced => 512,
            ModelTier::Quality => 768,
        }
    }
}

impl Default for ModelTier {
    fn default() -> Self {
        ModelTier::Balanced
    }
}

/// Entity extractor using native LLM inference
///
/// Uses Qwen3 model which produces valid JSON naturally.
pub struct LlmEntityExtractor {
    engine: InferenceEngine,
    tier: ModelTier,
}

impl LlmEntityExtractor {
    /// Load extractor with specified model tier
    ///
    /// Models are loaded from cache directory. If not present,
    /// returns an error with instructions to install the model crate.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use mnemefusion_core::extraction::{LlmEntityExtractor, ModelTier};
    ///
    /// let mut extractor = LlmEntityExtractor::load(ModelTier::Balanced)?;
    /// ```
    pub fn load(tier: ModelTier) -> Result<Self> {
        let model_path = Self::resolve_model_path(tier)?;
        let gpu_layers = InferenceEngine::detect_gpu_layers();

        let engine = InferenceEngine::load(&model_path, gpu_layers)?;

        Ok(Self { engine, tier })
    }

    /// Load extractor from a specific model path
    ///
    /// Use this when you have a custom model location.
    pub fn load_from_path(model_path: impl Into<PathBuf>, tier: ModelTier) -> Result<Self> {
        let model_path = model_path.into();
        let gpu_layers = InferenceEngine::detect_gpu_layers();

        let engine = InferenceEngine::load(&model_path, gpu_layers)?;

        Ok(Self { engine, tier })
    }

    /// Extract entity facts from text content
    ///
    /// When `speaker` is provided, the LLM prompt includes speaker context
    /// so that first-person statements ("I love hiking") are correctly
    /// attributed to the speaker entity rather than to objects mentioned.
    ///
    /// # Arguments
    ///
    /// * `content` - The text to extract entities and facts from
    /// * `speaker` - Optional name of who spoke this text (from conversation metadata)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = extractor.extract("Alice works at Google", None)?;
    /// // With speaker context:
    /// let result = extractor.extract("I'm researching adoption", Some("Caroline"))?;
    /// // Facts correctly attributed to Caroline, not to "adoption"
    /// ```
    pub fn extract(&self, content: &str, speaker: Option<&str>) -> Result<ExtractionResult> {
        if content.trim().is_empty() {
            return Ok(ExtractionResult::empty());
        }

        // Prepend speaker name to content so the model sees it in the text,
        // improving entity attribution for first-person speech.
        let attributed_content = if let Some(name) = speaker {
            format!("{} says: {}", name, content)
        } else {
            content.to_string()
        };

        let prompt = build_fewshot_extraction_prompt(&attributed_content, speaker);

        // Generate without grammar — grammar-constrained sampling crashes on some platforms.
        // We fix malformed JSON in post-processing instead.
        let raw_output = self.engine.generate(&prompt, self.tier.max_tokens())?;

        // Fix common JSON malformations from the model
        let fixed_output = Self::fix_json(&raw_output);

        // Extract JSON from output (may have extra text)
        let json_output = Self::extract_json(&fixed_output)?;

        // Parse and validate
        let mut result: ExtractionResult = serde_json::from_str(&json_output).map_err(|e| {
            Error::InferenceError(format!(
                "JSON parsing failed: {}. Output was: {}",
                e, fixed_output
            ))
        })?;

        // Post-process: fix entity attribution for first-person speech.
        // The model often generates entity="I"/"me"/"my" instead of the speaker name.
        if let Some(name) = speaker {
            let name_lower = name.to_lowercase();
            for fact in &mut result.entity_facts {
                let entity_lower = fact.entity.to_lowercase();
                if entity_lower == "i" || entity_lower == "me" || entity_lower == "my"
                    || entity_lower == "myself" || entity_lower == "the speaker"
                {
                    fact.entity = name.to_string();
                }
            }
            // Also fix entity list
            for entity in &mut result.entities {
                let entity_lower = entity.name.to_lowercase();
                if entity_lower == "i" || entity_lower == "me" || entity_lower == "my"
                    || entity_lower == "myself" || entity_lower == "the speaker"
                {
                    entity.name = name.to_string();
                }
            }
            // Ensure speaker is in entities list if they have facts
            let has_speaker_facts = result.entity_facts.iter().any(|f| f.entity.to_lowercase() == name_lower);
            let speaker_in_entities = result.entities.iter().any(|e| e.name.to_lowercase() == name_lower);
            if has_speaker_facts && !speaker_in_entities {
                result.entities.push(crate::extraction::output::ExtractedEntity {
                    name: name.to_string(),
                    entity_type: "person".to_string(),
                });
            }
        }

        Ok(result)
    }

    /// Fix common JSON malformations produced by the model
    ///
    /// Handles:
    /// - `0-0.8` → `0.8` (model generates dash between digits in numbers)
    /// - `{{` → `{` and `}}` → `}` (double braces from format string mimicry)
    /// - Trailing commas before `]` or `}`
    fn fix_json(output: &str) -> String {
        let mut fixed = output.to_string();

        // Fix double braces (model mimics Rust format string escaping)
        // Only fix if the output has `{{` patterns that aren't inside strings
        if fixed.contains("{{") && fixed.contains("}}") {
            fixed = fixed.replace("{{", "{").replace("}}", "}");
        }

        // Fix malformed numbers like `0-0.8` → `0.8` (model generates `0-` prefix)
        // Pattern: digit(s)-digit(s).digit(s) where the first part before `-` is spurious
        let mut result = String::with_capacity(fixed.len());
        let chars: Vec<char> = fixed.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            // Look for pattern: `:` ws `0-0.` or `:` ws `0-1`
            if i + 3 < chars.len() && chars[i].is_ascii_digit() && chars[i + 1] == '-' && chars[i + 2].is_ascii_digit() {
                // Check if this is inside a number context (after `:` or `,`)
                let prev_non_ws = result.trim_end().chars().last();
                if prev_non_ws == Some(':') || prev_non_ws == Some(',') {
                    // Skip the `0-` prefix, keep the rest of the number
                    i += 2; // skip the digit and dash
                    continue;
                }
            }
            result.push(chars[i]);
            i += 1;
        }

        // Fix trailing commas: `[..., ]` → `[...]` and `{..., }` → `{...}`
        result = result.replace(", ]", "]").replace(",]", "]")
                       .replace(", }", "}").replace(",}", "}");

        result
    }

    /// Extract JSON object from model output
    fn extract_json(output: &str) -> Result<String> {
        // Find the first { and last matching }
        let start = output.find('{').ok_or_else(|| {
            Error::InferenceError(format!("No JSON object found in output: {}", output))
        })?;

        // Find matching closing brace
        let mut depth = 0;
        let mut end = start;
        for (i, ch) in output[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i;
                        break;
                    }
                }
                _ => {}
            }
        }

        if depth != 0 {
            return Err(Error::InferenceError(format!(
                "Unbalanced braces in JSON output: {}",
                output
            )));
        }

        Ok(output[start..=end].to_string())
    }

    /// Reset the GPU context to prevent memory fragmentation during long ingestion runs.
    pub fn reset_context(&self) {
        self.engine.reset_context();
    }

    /// Get the model tier being used
    pub fn tier(&self) -> ModelTier {
        self.tier
    }

    /// Get the model name
    pub fn model_name(&self) -> String {
        self.engine.model_name()
    }

    /// Resolve model path from cache or environment
    fn resolve_model_path(tier: ModelTier) -> Result<PathBuf> {
        // 1. Check environment override
        if let Ok(path) = std::env::var("MNEMEFUSION_MODEL_PATH") {
            let path = PathBuf::from(path);
            if path.exists() {
                return Ok(path);
            }
        }

        // 2. Check project-local models directory (organized by tier)
        let tier_dir = match tier {
            ModelTier::Balanced => "qwen3-4b",
            ModelTier::Quality => "qwen3-8b",
        };
        let local_models = PathBuf::from("models").join(tier_dir).join(tier.model_filename());
        if local_models.exists() {
            return Ok(local_models);
        }

        // 3. Check project-local models directory (flat)
        let local_flat = PathBuf::from("models").join(tier.model_filename());
        if local_flat.exists() {
            return Ok(local_flat);
        }

        // 4. Check cache directory
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mnemefusion")
            .join("models");

        let model_path = cache_dir.join(tier.model_filename());

        if model_path.exists() {
            return Ok(model_path);
        }

        // 5. Model not found - return error with instructions
        Err(Error::ModelNotFound(format!(
            "Model {} not found. Please either:\n\
             1. Set MNEMEFUSION_MODEL_PATH environment variable to point to your GGUF model\n\
             2. Place the model in: models/{}/{}\n\
             3. Place the model in: {}\n\
             4. Add mnemefusion-model-{} to your dependencies (coming soon)",
            tier.model_filename(),
            tier_dir,
            tier.model_filename(),
            model_path.display(),
            match tier {
                ModelTier::Balanced => "4b",
                ModelTier::Quality => "8b",
            }
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_tier_default() {
        assert_eq!(ModelTier::default(), ModelTier::Balanced);
    }

    #[test]
    fn test_model_tier_filenames() {
        assert!(ModelTier::Balanced.model_filename().contains("4B"));
        assert!(ModelTier::Quality.model_filename().contains("8B"));
    }

    #[test]
    fn test_model_tier_gpu_layers() {
        assert!(ModelTier::Balanced.default_gpu_layers() > 0);
        assert!(ModelTier::Quality.default_gpu_layers() > 0);
    }

    #[test]
    fn test_resolve_model_path_env_override() {
        // Set environment variable to a non-existent path
        std::env::set_var("MNEMEFUSION_MODEL_PATH", "/nonexistent/model.gguf");

        // Should still check if the path exists and fall through
        let result = LlmEntityExtractor::resolve_model_path(ModelTier::Balanced);
        assert!(result.is_err()); // Path doesn't exist

        std::env::remove_var("MNEMEFUSION_MODEL_PATH");
    }

    #[test]
    fn test_model_not_found_error_message() {
        std::env::remove_var("MNEMEFUSION_MODEL_PATH");

        let result = LlmEntityExtractor::resolve_model_path(ModelTier::Balanced);
        assert!(result.is_err());

        let error = result.unwrap_err();
        let error_msg = error.to_string();
        assert!(error_msg.contains("not found"));
        assert!(error_msg.contains("MNEMEFUSION_MODEL_PATH"));
    }
}
