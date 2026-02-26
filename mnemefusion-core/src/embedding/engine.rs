//! ONNX-based embedding engine using fastembed-rs.
//!
//! Provides automatic text→vector conversion via the same model weights
//! as Python sentence-transformers (both use the HF-exported ONNX graph),
//! so embeddings are numerically compatible with existing databases.

use crate::{Error, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Embedding engine backed by fastembed (ONNX Runtime).
///
/// Produces 768-dim BGE-base-en-v1.5 embeddings compatible with existing
/// sentence-transformers databases — both use the same ONNX model weights.
pub struct EmbeddingEngine {
    model: TextEmbedding,
    pub dim: usize,
}

impl EmbeddingEngine {
    /// Initialize from a local model cache directory.
    ///
    /// `path` should be the root directory of a fastembed/HF-hub cache that
    /// contains the BGE-base-en-v1.5 model. Fastembed will skip the network
    /// download if the model files already exist in this directory.
    ///
    /// To pre-download: `python -c "from fastembed import TextEmbedding;
    /// TextEmbedding(['hello'])"` — model lands in `~/.cache/fastembed/`.
    ///
    /// # Errors
    /// Returns `Error::Configuration` if the model cannot be loaded.
    pub fn from_path(path: &str) -> Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGEBaseENV15)
                .with_cache_dir(std::path::PathBuf::from(path))
                .with_show_download_progress(false),
        )
        .map_err(|e| {
            Error::Configuration(format!(
                "Failed to load embedding model from '{}': {}",
                path, e
            ))
        })?;

        Ok(Self { model, dim: 768 })
    }

    /// Initialize by model ID with automatic download from HuggingFace Hub.
    ///
    /// Currently supports `"BAAI/bge-base-en-v1.5"` (768-dim).
    /// Model is cached in the default fastembed directory (`~/.cache/fastembed/`).
    ///
    /// # Errors
    /// Returns `Error::Configuration` if the download or load fails.
    pub fn from_model_id(_model_id: &str) -> Result<Self> {
        // Only BGE-base supported for now — matches existing DB embedding dim.
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGEBaseENV15))
            .map_err(|e| {
                Error::Configuration(format!("Failed to load embedding model: {}", e))
            })?;

        Ok(Self { model, dim: 768 })
    }

    /// Embed a single string. Returns a 768-dim float vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self
            .model
            .embed(vec![text], None)
            .map_err(|e| Error::Configuration(format!("Embedding failed: {}", e)))?;

        results
            .pop()
            .ok_or_else(|| Error::Configuration("Empty embedding result".to_string()))
    }

    /// Embed a batch of strings. More efficient than calling `embed` repeatedly.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed(texts.to_vec(), None)
            .map_err(|e| Error::Configuration(format!("Batch embedding failed: {}", e)))
    }
}

impl std::fmt::Debug for EmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EmbeddingEngine {{ dim: {} }}", self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require the BGE model files to be available.
    // Run with: cargo test --features embedding-onnx -- embedding
    // They are ignored by default to avoid network access during CI.

    #[test]
    #[ignore = "requires BGE model download"]
    fn test_embedding_engine_from_model_id() {
        let engine = EmbeddingEngine::from_model_id("BAAI/bge-base-en-v1.5").unwrap();
        assert_eq!(engine.dim, 768);
    }

    #[test]
    #[ignore = "requires BGE model download"]
    fn test_embedding_engine_embed_dim() {
        let engine = EmbeddingEngine::from_model_id("BAAI/bge-base-en-v1.5").unwrap();
        let v = engine.embed("hello world").unwrap();
        assert_eq!(v.len(), 768);
    }

    #[test]
    #[ignore = "requires BGE model download"]
    fn test_embedding_engine_parity() {
        let engine = EmbeddingEngine::from_model_id("BAAI/bge-base-en-v1.5").unwrap();
        let v1 = engine.embed("hello world").unwrap();
        let v2 = engine.embed("hello world").unwrap();

        // Same text → cosine similarity ≈ 1.0
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm1 * norm2);
        assert!(
            (cosine - 1.0).abs() < 1e-5,
            "Expected cosine ≈ 1.0, got {}",
            cosine
        );
    }

    #[test]
    #[ignore = "requires BGE model download"]
    fn test_embedding_engine_batch() {
        let engine = EmbeddingEngine::from_model_id("BAAI/bge-base-en-v1.5").unwrap();
        let results = engine.embed_batch(&["hello", "world"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 768);
        assert_eq!(results[1].len(), 768);
    }
}
