//! LLM inference engine using llama.cpp
//!
//! Provides native Rust inference with GPU acceleration and grammar-constrained decoding.

use crate::error::{Error, Result};
use crate::inference::JsonGrammar;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

/// Parameters for controlling LLM text generation
///
/// Different parameter combinations produce diverse extraction results
/// while maintaining JSON validity. Used by `generate_with_params()`.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Sampling temperature (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Random seed for reproducibility
    pub seed: u32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            top_p: 0.9,
            seed: 42,
        }
    }
}

/// Process-global llama backend. Initialized once, lives for the entire process.
/// This prevents `BackendAlreadyInitialized` errors when creating multiple
/// `InferenceEngine` instances within the same process.
static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Native LLM inference engine with grammar-constrained decoding
///
/// Reuses a single GPU context across generate() calls to prevent
/// GPU memory fragmentation from repeated allocation/deallocation.
pub struct InferenceEngine {
    // SAFETY: ctx borrows from model via Arc. Declared before model so it's
    // dropped first. Lifetime erased via transmute since Arc guarantees the
    // model outlives the engine. See get_or_create_ctx().
    ctx: Mutex<Option<LlamaContext<'static>>>,
    model: Arc<LlamaModel>,
    n_ctx: u32,
}

// SAFETY: LlamaContext contains NonNull<llama_context> which is !Send, but:
// 1. The context is behind a Mutex, ensuring exclusive access
// 2. llama.cpp contexts are safe to move between threads when not in active use
// 3. PyO3's #[pyclass] requires Send for the containing MemoryEngine
unsafe impl Send for InferenceEngine {}

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

        // Initialize llama.cpp backend once (process-global).
        // Subsequent calls reuse the existing backend, avoiding BackendAlreadyInitialized.
        let backend = LLAMA_BACKEND.get_or_init(|| {
            let b = LlamaBackend::init().expect("Failed to initialize llama backend");
            // Load ggml backends as dynamic libraries on first init.
            // When gpu_layers=0, skip CUDA to avoid allocating GPU compute buffers
            // (which can fail on low-VRAM cards and waste memory even when not needed).
            Self::load_ggml_backends(gpu_layers > 0);
            b
        });

        // Configure model parameters
        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);

        // Load the model
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| Error::InferenceError(format!("Model load: {e}")))?;

        let n_ctx = 2048; // Context size for entity extraction

        Ok(Self {
            ctx: Mutex::new(None),
            model: Arc::new(model),
            n_ctx,
        })
    }

    /// Get or create a reusable context, clearing KV cache between calls.
    ///
    /// On first call, allocates a GPU context. On subsequent calls, reuses it
    /// with a KV cache clear — avoiding the GPU memory fragmentation that
    /// causes crashes after ~400 allocation/deallocation cycles.
    ///
    /// # Safety
    /// The returned `&mut LlamaContext<'static>` is transmuted from `LlamaContext<'model>`.
    /// This is sound because:
    /// - `model` is behind `Arc`, guaranteed to outlive any context usage
    /// - `ctx` is declared before `model` in the struct, so it's dropped first
    /// - The C library context holds a raw pointer internally, not a Rust reference
    fn get_or_create_ctx(&self) -> Result<std::sync::MutexGuard<'_, Option<LlamaContext<'static>>>> {
        let mut guard = self.ctx.lock().unwrap();
        if guard.is_some() {
            guard.as_mut().unwrap().clear_kv_cache();
        } else {
            let n_batch = 512u32;
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(self.n_ctx))
                .with_n_batch(n_batch);

            let backend = LLAMA_BACKEND.get()
                .ok_or_else(|| Error::InferenceError("Backend not initialized".to_string()))?;
            let new_ctx = self
                .model
                .new_context(backend, ctx_params)
                .map_err(|e| Error::InferenceError(format!("Context creation: {e}")))?;

            // SAFETY: model is behind Arc, guaranteed to live as long as InferenceEngine.
            // See struct-level safety comment.
            let static_ctx: LlamaContext<'static> = unsafe { std::mem::transmute(new_ctx) };
            *guard = Some(static_ctx);
        }
        Ok(guard)
    }

    /// Drop and reset the GPU context to prevent memory fragmentation.
    ///
    /// After ~400 inference cycles, the GPU allocator can become fragmented,
    /// causing crashes. Calling this periodically (e.g., every 200 docs)
    /// forces a fresh context allocation on the next generate() call.
    pub fn reset_context(&self) {
        let mut guard = self.ctx.lock().unwrap();
        *guard = None;
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
        let n_batch = 512usize;
        let mut ctx_guard = self.get_or_create_ctx()?;
        let ctx = ctx_guard.as_mut().unwrap();

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Error::InferenceError(format!("Tokenization: {e}")))?;

        // Process prompt in chunks of n_batch (prompt may exceed n_batch)
        let mut batch = LlamaBatch::new(n_batch, 1);
        let mut i = 0;
        while i < tokens.len() {
            batch.clear();
            let chunk_end = std::cmp::min(i + n_batch, tokens.len());
            for j in i..chunk_end {
                let is_last = j == tokens.len() - 1;
                batch
                    .add(tokens[j], j as i32, &[0], is_last)
                    .map_err(|e| Error::InferenceError(format!("Batch add: {e}")))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| Error::InferenceError(format!("Prompt decode: {e}")))?;
            i = chunk_end;
        }

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
        let output = self.detokenize_lossy(&output_tokens)?;

        Ok(output)
    }

    /// Generate without grammar constraints (for testing)
    pub fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String> {
        let n_batch = 512usize;
        let mut ctx_guard = self.get_or_create_ctx()?;
        let ctx = ctx_guard.as_mut().unwrap();

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Error::InferenceError(format!("Tokenization: {e}")))?;

        // Process prompt in chunks of n_batch (prompt may exceed n_batch)
        let mut batch = LlamaBatch::new(n_batch, 1);
        let mut i = 0;
        while i < tokens.len() {
            batch.clear();
            let chunk_end = std::cmp::min(i + n_batch, tokens.len());
            for j in i..chunk_end {
                let is_last = j == tokens.len() - 1;
                batch
                    .add(tokens[j], j as i32, &[0], is_last)
                    .map_err(|e| Error::InferenceError(format!("Batch add: {e}")))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| Error::InferenceError(format!("Prompt decode: {e}")))?;
            i = chunk_end;
        }

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

        let output = self.detokenize_lossy(&output_tokens)?;

        Ok(output)
    }

    /// Generate without grammar constraints using custom sampling parameters
    ///
    /// Same as `generate()` but with configurable temperature, top_p, and seed.
    /// Used by multi-pass extraction to produce diverse outputs across passes.
    pub fn generate_with_params(
        &self,
        prompt: &str,
        max_tokens: u32,
        params: &GenerationParams,
    ) -> Result<String> {
        let n_batch = 512usize;
        let mut ctx_guard = self.get_or_create_ctx()?;
        let ctx = ctx_guard.as_mut().unwrap();

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Error::InferenceError(format!("Tokenization: {e}")))?;

        // Process prompt in chunks of n_batch
        let mut batch = LlamaBatch::new(n_batch, 1);
        let mut i = 0;
        while i < tokens.len() {
            batch.clear();
            let chunk_end = std::cmp::min(i + n_batch, tokens.len());
            for j in i..chunk_end {
                let is_last = j == tokens.len() - 1;
                batch
                    .add(tokens[j], j as i32, &[0], is_last)
                    .map_err(|e| Error::InferenceError(format!("Batch add: {e}")))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| Error::InferenceError(format!("Prompt decode: {e}")))?;
            i = chunk_end;
        }

        // Create sampler with custom parameters
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::dist(params.seed),
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

        let output = self.detokenize_lossy(&output_tokens)?;

        Ok(output)
    }

    /// Convert tokens to string with lossy UTF-8 handling.
    ///
    /// Some models (notably Phi-4-mini) produce token sequences where individual
    /// token bytes form valid UTF-8 only when concatenated. The bulk `tokens_to_str()`
    /// method fails with `FromUtf8Error` in these cases. This method converts each
    /// token to raw bytes, concatenates them, then uses lossy UTF-8 conversion.
    fn detokenize_lossy(&self, tokens: &[LlamaToken]) -> Result<String> {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for &token in tokens {
            match self
                .model
                .token_to_bytes(token, llama_cpp_2::model::Special::Tokenize)
            {
                Ok(token_bytes) => bytes.extend_from_slice(&token_bytes),
                Err(e) => {
                    tracing::warn!("Failed to decode token {}: {}", token.0, e);
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Load ggml backend DLLs (CPU + CUDA) from known locations
    fn load_ggml_backends(load_cuda: bool) {
        use std::ffi::CString;

        // Backend shared libraries to load. CPU is always required.
        // CUDA is skipped when gpu_layers=0 to avoid allocating VRAM compute buffers
        // (~600MB) that aren't needed for pure CPU inference.
        let ext = if cfg!(target_os = "windows") { "dll" } else { "so" };
        let cpu_name = format!("ggml-cpu.{}", ext);
        let cuda_name = format!("ggml-cuda.{}", ext);
        let dll_names: Vec<String> = if load_cuda {
            vec![cpu_name, cuda_name]
        } else {
            tracing::info!("GPU layers = 0, skipping CUDA backend (CPU-only mode)");
            vec![cpu_name]
        };

        // Search locations for ggml backend shared libraries.
        // Order: explicit env var → near the compiled library → build output → system paths.
        let mut search_paths: Vec<std::path::PathBuf> = Vec::new();

        // 1. MNEMEFUSION_DLL_DIR env var (highest priority, user override)
        if let Ok(dir) = std::env::var("MNEMEFUSION_DLL_DIR") {
            search_paths.push(std::path::PathBuf::from(dir));
        }

        // 2. Current working directory
        if let Ok(cwd) = std::env::current_dir() {
            search_paths.push(cwd);
        }

        // 3. Directory of the current executable
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                search_paths.push(dir.to_path_buf());
            }
        }

        // 4. Workspace root (compile-time: CARGO_MANIFEST_DIR parent)
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if let Some(workspace) = manifest_dir.parent() {
            search_paths.push(workspace.to_path_buf());
            // Also check target/release where build artifacts go
            let release_dir = workspace.join("target").join("release");
            if release_dir.exists() {
                search_paths.push(release_dir);
            }
        }

        // 5. Auto-detect from llama-cpp-sys-2 build output directory.
        // cmake places built backends in target/{profile}/build/llama-cpp-sys-2-*/out/build/bin/
        // On Windows they're in a Release/ subdirectory.
        if let Some(workspace) = manifest_dir.parent() {
            for profile in &["release", "debug"] {
                let build_dir = workspace.join("target").join(profile).join("build");
                if build_dir.exists() {
                    if let Ok(entries) = std::fs::read_dir(&build_dir) {
                        for entry in entries.flatten() {
                            let name = entry.file_name();
                            let name_str = name.to_string_lossy();
                            if name_str.starts_with("llama-cpp-sys-2-") {
                                let bin_dir = entry.path().join("out").join("build").join("bin");
                                if bin_dir.exists() {
                                    // Windows: cmake puts DLLs in bin/Release/
                                    let release_sub = bin_dir.join("Release");
                                    if release_sub.exists() {
                                        search_paths.push(release_sub);
                                    }
                                    search_paths.push(bin_dir);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 6. mnemefusion.libs/ directory (maturin Linux wheel bundling).
        // maturin uses patchelf to bundle shared libs here on Linux pip install.
        if let Some(workspace) = manifest_dir.parent() {
            let libs_dir = workspace.join("mnemefusion.libs");
            if libs_dir.exists() {
                search_paths.push(libs_dir);
            }
        }
        // Also check relative to CWD (for pip-installed packages run from site-packages)
        if let Ok(cwd) = std::env::current_dir() {
            let libs_dir = cwd.join("mnemefusion.libs");
            if libs_dir.exists() {
                search_paths.push(libs_dir);
            }
        }

        // 7. LD_LIBRARY_PATH / PATH directories (system library search paths)
        let path_var = if cfg!(target_os = "windows") { "PATH" } else { "LD_LIBRARY_PATH" };
        if let Ok(paths) = std::env::var(path_var) {
            for p in std::env::split_paths(&paths) {
                if p.exists() {
                    search_paths.push(p);
                }
            }
        }

        for dll_name in &dll_names {
            let mut loaded = false;
            // On Linux, cmake produces "libggml-cpu.so" (with lib prefix).
            // Try both the bare name and the lib-prefixed name.
            let lib_prefixed = format!("lib{}", dll_name);
            let candidates = [dll_name.as_str(), lib_prefixed.as_str()];
            for dir in &search_paths {
                for candidate in &candidates {
                    let dll_path = dir.join(candidate);
                    if dll_path.exists() {
                        if let Ok(path_str) = CString::new(dll_path.to_string_lossy().as_bytes()) {
                            let reg = unsafe {
                                llama_cpp_sys_2::ggml_backend_load(path_str.as_ptr())
                            };
                            if !reg.is_null() {
                                tracing::info!("Loaded backend: {} from {}", dll_name, dll_path.display());
                                loaded = true;
                                break;
                            } else {
                                tracing::warn!("Found but failed to load: {}", dll_path.display());
                            }
                        }
                    }
                }
                if loaded { break; }
            }
            if !loaded {
                let dirs: Vec<String> = search_paths.iter().map(|p| p.display().to_string()).collect();
                tracing::warn!(
                    "Backend not found: {} (searched {} dirs: {})",
                    dll_name,
                    search_paths.len(),
                    dirs.join(", ")
                );
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
