//! Embedding model integration for automatic text vectorization.
//!
//! When the `embedding-onnx` feature is enabled, provides `EmbeddingEngine`
//! which wraps `fastembed` (ONNX Runtime) to compute embeddings internally.
//! This removes the need for callers to supply embedding vectors explicitly.

#[cfg(feature = "embedding-onnx")]
pub mod engine;

#[cfg(feature = "embedding-onnx")]
pub use engine::EmbeddingEngine;
