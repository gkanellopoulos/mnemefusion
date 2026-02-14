//! Query module for intent classification and multi-dimensional retrieval
//!
//! This module provides intelligent query planning across all dimensions:
//! - Intent classification (temporal, causal, entity, factual)
//! - Adaptive weight selection based on intent
//! - Multi-dimensional result fusion
//! - Coordinated retrieval across semantic, temporal, causal, and entity indexes
//! - Multi-turn aggregation for list/collection queries
//! - Policy-guided graph traversal for retrieval expansion

pub mod aggregator;
pub mod fusion;
pub mod graph_traversal;
pub mod intent;
pub mod planner;
pub mod profile_search;
pub mod reranker;

pub use aggregator::{MultiTurnAggregator, QueryType};
pub use fusion::{
    AdaptiveWeightConfig, FusedResult, FusionEngine, FusionStrategy, IntentWeights,
};
pub use graph_traversal::{GraphTraversal, TraversalConfig};
pub use intent::{IntentClassification, IntentClassifier, QueryIntent};
pub use planner::QueryPlanner;
pub use profile_search::{MatchedProfileFact, ProfileSearch, ProfileSearchResult};
pub use reranker::HeuristicReranker;
