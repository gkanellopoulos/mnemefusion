/// Configuration for Small Language Model intent classification
///
/// This module provides configuration types for the SLM-based intent classifier.
/// When enabled, the SLM classifier provides semantic understanding of query intent,
/// improving classification accuracy from pattern-based ~30% to target 85%+.
///
/// The SLM is optional and disabled by default. Pattern-based classification remains
/// the fallback to ensure zero regression if SLM fails or is not configured.
use std::path::PathBuf;

/// Configuration for SLM-based intent classification
#[derive(Debug, Clone)]
pub struct SlmConfig {
    /// HuggingFace model ID
    ///
    /// Default: "google/gemma-2-2b-it" (Instruction-tuned Gemma 2 2B)
    ///
    /// Used for automatic downloads. If you manually placed model files,
    /// this ID should match the directory name in your model_path.
    pub model_id: String,

    /// Optional path to locally stored model files
    ///
    /// If provided, the library will load model files from this directory
    /// instead of downloading from HuggingFace Hub.
    ///
    /// The directory should contain:
    /// - model.safetensors (or model-*.safetensors files)
    /// - config.json
    /// - tokenizer.json
    ///
    /// Example:
    /// ```
    /// let config = SlmConfig::default()
    ///     .with_model_path("/path/to/gemma-2-2b-it");
    /// ```
    pub model_path: Option<PathBuf>,

    /// HuggingFace API token for accessing gated models
    ///
    /// Only needed if you want automatic downloads from HuggingFace.
    /// Not required if you manually place model files using `model_path`.
    ///
    /// Default: None (will check HF_TOKEN environment variable)
    pub hf_token: Option<String>,

    /// Local cache directory for model weights (when downloading)
    ///
    /// Default: ~/.cache/mnemefusion/models
    /// Only used for automatic downloads. Ignored if model_path is set.
    pub cache_dir: PathBuf,

    /// Maximum inference timeout in milliseconds
    ///
    /// Default: 100ms
    /// If inference takes longer than this, the classifier will fall back to patterns.
    pub timeout_ms: u64,

    /// Enable GPU acceleration
    ///
    /// Default: false (CPU only)
    /// GPU support can be enabled with ONNX Runtime.
    pub use_gpu: bool,

    /// Minimum confidence threshold for SLM classification
    ///
    /// Default: 0.6
    /// If SLM confidence is below this threshold, fall back to pattern classifier.
    pub min_confidence: f32,
}

impl Default for SlmConfig {
    fn default() -> Self {
        // Get cache directory, falling back to current directory if not available
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("mnemefusion")
            .join("models");

        // Try to get HF token from environment variable
        let hf_token = std::env::var("HF_TOKEN").ok();

        // Try to get model path from MODEL_PATH environment variable
        let model_path = std::env::var("MODEL_PATH").ok().map(PathBuf::from);

        Self {
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(), // Qwen3 0.6B - reasoning optimized
            model_path, // Read from MODEL_PATH env var if available
            hf_token,
            cache_dir,
            timeout_ms: 100,
            use_gpu: false,
            min_confidence: 0.6,
        }
    }
}

impl SlmConfig {
    /// Create a new SLM configuration with the specified model ID
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Set the local model path
    ///
    /// Use this to load models from a local directory instead of downloading
    /// from HuggingFace. The directory must contain model.safetensors,
    /// config.json, and tokenizer.json.
    ///
    /// # Example
    /// ```no_run
    /// use mnemefusion_core::slm::SlmConfig;
    ///
    /// let config = SlmConfig::default()
    ///     .with_model_path("/models/gemma-2-2b-it");
    /// ```
    pub fn with_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the HuggingFace API token
    ///
    /// Only needed for automatic downloads. Not required if using manual model placement.
    /// Get your token at: https://huggingface.co/settings/tokens
    ///
    /// # Example
    /// ```no_run
    /// use mnemefusion_core::slm::SlmConfig;
    ///
    /// let config = SlmConfig::default()
    ///     .with_hf_token("hf_...");
    /// ```
    pub fn with_hf_token(mut self, token: impl Into<String>) -> Self {
        self.hf_token = Some(token.into());
        self
    }

    /// Set the cache directory
    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }

    /// Set the inference timeout in milliseconds
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Enable or disable GPU acceleration
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set the minimum confidence threshold
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SlmConfig::default();
        assert_eq!(config.model_id, "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(config.timeout_ms, 100);
        assert!(!config.use_gpu);
        assert_eq!(config.min_confidence, 0.6);
    }

    #[test]
    fn test_builder_pattern() {
        let config = SlmConfig::new("custom/model")
            .with_timeout_ms(200)
            .with_gpu(true)
            .with_min_confidence(0.8);

        assert_eq!(config.model_id, "custom/model");
        assert_eq!(config.timeout_ms, 200);
        assert!(config.use_gpu);
        assert_eq!(config.min_confidence, 0.8);
    }

    #[test]
    fn test_cache_dir_creation() {
        let config = SlmConfig::default();
        assert!(config.cache_dir.to_string_lossy().contains("mnemefusion"));
        assert!(config.cache_dir.to_string_lossy().contains("models"));
    }
}
