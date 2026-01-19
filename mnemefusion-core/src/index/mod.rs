//! Index layer - vector and temporal indexing
//!
//! This module provides indexing across different dimensions:
//! - Vector index (semantic similarity via HNSW)
//! - Temporal index (time-based retrieval)

pub mod temporal;
pub mod vector;

pub use temporal::{TemporalIndex, TemporalResult};
pub use vector::{VectorIndex, VectorIndexConfig, VectorResult};
