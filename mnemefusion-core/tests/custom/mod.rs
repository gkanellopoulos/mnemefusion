//! Custom test cases for MnemeFusion differentiators
//!
//! This module contains hand-crafted test cases for features that lack
//! standard benchmarks: temporal queries, causal reasoning, entity search,
//! intent classification, and adaptive fusion.
//!
//! Total: ~180 custom test cases
//! - Temporal: 50 cases
//! - Causal: 60 cases
//! - Entity: 35 cases
//! - Intent: 25 cases
//! - Fusion: 10 cases

mod test_utils;

// Re-export test utilities for use in test modules
pub use test_utils::*;

#[cfg(test)]
mod temporal_tests;

#[cfg(test)]
mod causal_tests;

#[cfg(test)]
mod entity_tests;

#[cfg(test)]
mod intent_tests;

#[cfg(test)]
mod fusion_tests;
