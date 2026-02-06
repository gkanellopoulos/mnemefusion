//! LLM inference engine using llama.cpp
//!
//! Provides native Rust inference with GPU acceleration and grammar-constrained decoding.

use crate::error::{Error, Result};
use crate::inference::JsonGrammar;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

/// Native LLM inference engine with grammar-constrained decoding
pub struct InferenceEngine {
    backend: LlamaBackend,
    model: Arc<LlamaModel>,
    n_ctx: u32,
}

impl InferenceEngine {
    /// Load model from a GGUF file
    ///
    /// # Arguments
    /// * `model_path` - Path to the GGUF model file
    /// * `gpu_layers` - Number of layers to offload to GPU (0 = CPU only, use large value for all GPU)
    ///
    /// # Example
    /// ```rust,ignore
    /// let engine = InferenceEngine::load("qwen3.5-4b.gguf", 99)?;
    /// ```
    pub fn load(model_path: impl AsRef<Path>, gpu_layers: u32) -> Result<Self> {
        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        // Initialize llama.cpp backend
        let backend =
            LlamaBackend::init().map_err(|e| Error::InferenceError(format!("Backend init: {e}")))?;

        // Load ggml backends (CPU + CUDA) as dynamic libraries
        Self::load_ggml_backends();

        // Configure model parameters
        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);

        // Load the model
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| Error::InferenceError(format!("Model load: {e}")))?;

        let n_ctx = 2048; // Context size for entity extraction

        Ok(Self {
            backend,
            model: Arc::new(model),
            n_ctx,
        })
    }

    /// Generate text with JSON grammar constraints
    ///
    /// The grammar ensures the output is always valid JSON matching the schema.
    pub fn generate_with_grammar(
        &self,
        prompt: &str,
        grammar: &JsonGrammar,
        max_tokens: u32,
    ) -> Result<String> {
        // Create context for this generation
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(512);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::InferenceError(format!("Context creation: {e}")))?;

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Error::InferenceError(format!("Tokenization: {e}")))?;

        // Create batch with prompt tokens
        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| Error::InferenceError(format!("Batch add: {e}")))?;
        }

        // Process the prompt
        ctx.decode(&mut batch)
            .map_err(|e| Error::InferenceError(format!("Prompt decode: {e}")))?;

        // Create sampler with grammar constraints
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.1),    // Low temperature for deterministic output
            LlamaSampler::top_p(0.9, 1), // Top-p sampling with min_keep=1
            LlamaSampler::dist(42),     // Seed for reproducibility
            LlamaSampler::grammar(&self.model, grammar.as_str(), "root")
                .map_err(|e| Error::InferenceError(format!("Grammar init: {e}")))?,
        ]);

        // Generate tokens
        let mut output_tokens: Vec<LlamaToken> = Vec::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            // Sample next token with grammar constraints
            let new_token = sampler.sample(&ctx, -1);

            // Check for end of generation
            if self.model.is_eog_token(new_token) {
                break;
            }

            output_tokens.push(new_token);

            // Prepare batch for next token
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| Error::InferenceError(format!("Batch add: {e}")))?;

            n_cur += 1;

            // Decode the new token
            ctx.decode(&mut batch)
                .map_err(|e| Error::InferenceError(format!("Token decode: {e}")))?;
        }

        // Convert tokens to string
        let output = self
            .model
            .tokens_to_str(&output_tokens, llama_cpp_2::model::Special::Tokenize)
            .map_err(|e| Error::InferenceError(format!("Detokenization: {e}")))?;

        Ok(output)
    }

    /// Generate without grammar constraints (for testing)
    pub fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String> {
        // Create context for this generation
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(512);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::InferenceError(format!("Context creation: {e}")))?;

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Error::InferenceError(format!("Tokenization: {e}")))?;

        // Create batch with prompt tokens
        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| Error::InferenceError(format!("Batch add: {e}")))?;
        }

        // Process the prompt
        ctx.decode(&mut batch)
            .map_err(|e| Error::InferenceError(format!("Prompt decode: {e}")))?;

        // Create sampler without grammar
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.1),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::dist(42),
        ]);

        // Generate tokens
        let mut output_tokens: Vec<LlamaToken> = Vec::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, -1);

            if self.model.is_eog_token(new_token) {
                break;
            }

            output_tokens.push(new_token);

            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| Error::InferenceError(format!("Batch add: {e}")))?;

            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| Error::InferenceError(format!("Token decode: {e}")))?;
        }

        let output = self
            .model
            .tokens_to_str(&output_tokens, llama_cpp_2::model::Special::Tokenize)
            .map_err(|e| Error::InferenceError(format!("Detokenization: {e}")))?;

        Ok(output)
    }

    /// Load ggml backend DLLs (CPU + CUDA) from known locations
    fn load_ggml_backends() {
        use std::ffi::CString;

        // Backend DLLs to load, in order (CPU first, then CUDA)
        let dll_names = ["ggml-cpu.dll", "ggml-cuda.dll"];

        // Search locations for ggml backend DLLs
        let mut search_paths: Vec<std::path::PathBuf> = Vec::new();

        // MNEMEFUSION_DLL_DIR env var (highest priority, user override)
        if let Ok(dir) = std::env::var("MNEMEFUSION_DLL_DIR") {
            search_paths.push(std::path::PathBuf::from(dir));
        }

        // Current working directory
        if let Ok(cwd) = std::env::current_dir() {
            search_paths.push(cwd);
        }

        // Directory of the current executable
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                search_paths.push(dir.to_path_buf());
            }
        }

        // Workspace root (compile-time: CARGO_MANIFEST_DIR parent)
        // DLLs are typically placed in the workspace root during development
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if let Some(workspace) = manifest_dir.parent() {
            search_paths.push(workspace.to_path_buf());
            // Also check target/release where build artifacts go
            let release_dir = workspace.join("target").join("release");
            if release_dir.exists() {
                search_paths.push(release_dir);
            }
        }

        for dll_name in &dll_names {
            let mut loaded = false;
            for dir in &search_paths {
                let dll_path = dir.join(dll_name);
                if dll_path.exists() {
                    if let Ok(path_str) = CString::new(dll_path.to_string_lossy().as_bytes()) {
                        let reg = unsafe {
                            llama_cpp_sys_2::ggml_backend_load(path_str.as_ptr())
                        };
                        if !reg.is_null() {
                            eprintln!("[mnemefusion] Loaded backend: {} from {}", dll_name, dll_path.display());
                            loaded = true;
                            break;
                        } else {
                            eprintln!("[mnemefusion] Found but failed to load: {}", dll_path.display());
                        }
                    }
                }
            }
            if !loaded {
                eprintln!("[mnemefusion] Backend not found: {}", dll_name);
            }
        }
    }

    /// Get model metadata
    pub fn model_name(&self) -> String {
        // Return model path or name if available
        "Qwen3.5".to_string()
    }

    /// Detect GPU availability and return recommended layer count
    ///
    /// Returns a high value (99) to use all layers on GPU, or 0 for CPU only.
    pub fn detect_gpu_layers() -> u32 {
        // Check environment override
        if let Ok(layers) = std::env::var("MNEMEFUSION_GPU_LAYERS") {
            if let Ok(n) = layers.parse() {
                return n;
            }
        }

        // Default: try to use GPU (99 = more than any model has, so all layers)
        // llama.cpp will fall back to CPU if GPU not available
        99
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gpu_layers_default() {
        std::env::remove_var("MNEMEFUSION_GPU_LAYERS");
        assert_eq!(InferenceEngine::detect_gpu_layers(), 99);
    }

    #[test]
    fn test_detect_gpu_layers_env_override() {
        std::env::set_var("MNEMEFUSION_GPU_LAYERS", "10");
        assert_eq!(InferenceEngine::detect_gpu_layers(), 10);
        std::env::remove_var("MNEMEFUSION_GPU_LAYERS");
    }
}
