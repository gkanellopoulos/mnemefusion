//! Graph module for causal and entity relationships.
//!
//! This module provides graph-based indexing for:
//! - Causal relationships between memories (cause → effect)
//! - Entity-memory relationships (future Sprint 5)

pub mod causal;
pub(crate) mod persist;

pub use causal::{CausalEdge, CausalPath, CausalTraversalResult, GraphManager};
