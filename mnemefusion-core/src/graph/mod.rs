//! Graph module for causal and entity relationships.
//!
//! This module provides graph-based indexing for:
//! - Causal relationships between memories (cause → effect)
//! - Entity-memory relationships

pub mod causal;
pub mod entity;
pub(crate) mod persist;

pub use causal::{CausalEdge, CausalPath, CausalTraversalResult, GraphManager};
pub use entity::{EntityEdge, EntityGraph, EntityNode, EntityQueryResult};
