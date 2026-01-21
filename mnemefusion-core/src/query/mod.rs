//! Query module for intent classification and multi-dimensional retrieval
//!
//! This module provides intelligent query planning across all dimensions:
//! - Intent classification (temporal, causal, entity, factual)
//! - Adaptive weight selection based on intent
//! - Multi-dimensional result fusion
//! - Coordinated retrieval across semantic, temporal, causal, and entity indexes

pub mod fusion;
pub mod intent;
pub mod planner;

pub use fusion::{AdaptiveWeightConfig, FusedResult, FusionEngine, IntentWeights};
pub use intent::{IntentClassification, IntentClassifier, QueryIntent};
pub use planner::QueryPlanner;
