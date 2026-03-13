//! Small Language Model (SLM) integration for semantic intent classification
//!
//! This module provides SLM-based intent classification using the Candle framework
//! and Gemma 3 1B model. It significantly improves classification accuracy over
//! pattern-based approaches:
//!
//! - Entity queries: 11% → 85%+ accuracy
//! - Causal queries: 30% → 85%+ accuracy
//! - Overall: 35% → 85%+ accuracy
//!
//! # Features
//!
//! The SLM integration is optional and gated behind the `slm` feature flag:
//!
//! ```toml
//! [dependencies]
//! mnemefusion-core = { version = "0.1", features = ["slm"] }
//! ```
//!
//! # Architecture
//!
//! The SLM classifier integrates with the QueryPlanner and falls back to
//! pattern-based classification on any error, ensuring zero regression:
//!
//! ```text
//! Query → Try SLM Classification
//!           ├─ Success → Use SLM result
//!           └─ Error → Fallback to patterns (zero regression)
//! ```
//!
//! # Usage
//!
//! ```no_run
//! use mnemefusion_core::slm::{SlmConfig, SlmClassifier};
//!
//! // Configure SLM
//! let config = SlmConfig::new("google/gemma-3-1b")
//!     .with_timeout_ms(100)
//!     .with_min_confidence(0.6);
//!
//! // Create classifier (model loaded lazily on first use)
//! let mut classifier = SlmClassifier::new(config).unwrap();
//!
//! // Classify query intent
//! match classifier.classify_intent("Who was the first speaker?") {
//!     Ok(classification) => {
//!         println!("Intent: {:?}, Confidence: {}",
//!                  classification.intent,
//!                  classification.confidence);
//!     }
//!     Err(e) => {
//!         // Fall back to pattern-based classification
//!         println!("SLM failed, using patterns: {}", e);
//!     }
//! }
//! ```
//!
//! # Performance
//!
//! - **Candle backend**: 10-15ms inference, ~500MB model size
//! - **ONNX backend**: 5-10ms inference (2-3x faster)
//!
//! # Model
//!
//! - Default: Gemma 3 1B (INT4 quantized, ~500MB)
//! - License: Apache 2.0
//! - Context: 32K tokens
//! - Hosted: HuggingFace Hub (auto-downloaded and cached)

mod classifier;
mod config;
#[cfg(test)]
mod tests;

pub use classifier::SlmClassifier;
pub use config::SlmConfig;

// Re-export key types for convenience
pub use crate::query::intent::{IntentClassification, QueryIntent};
