//! Index layer - vector, temporal, and keyword indexing
//!
//! This module provides indexing across different dimensions:
//! - Vector index (semantic similarity via HNSW)
//! - Temporal index (time-based retrieval)
//! - BM25 index (keyword-based retrieval)

pub mod bm25;
pub mod temporal;
pub mod vector;

pub use bm25::{BM25Config, BM25Index, BM25Result};
pub use temporal::{TemporalIndex, TemporalResult};
pub use vector::{VectorIndex, VectorIndexConfig, VectorResult};
