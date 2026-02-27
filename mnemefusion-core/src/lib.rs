//! MnemeFusion Core
//!
//! A unified memory engine for AI applications—"SQLite for AI memory."
//!
//! MnemeFusion provides four-dimensional memory indexing (semantic, temporal,
//! causal, entity) in a single embedded database file with zero external dependencies.
//!
//! # Quick Start
//!
//! ```no_run
//! use mnemefusion_core::{MemoryEngine, Config};
//!
//! // Open or create a database
//! let engine = MemoryEngine::open("./brain.mfdb", Config::default()).unwrap();
//!
//! // Add a memory
//! let embedding = vec![0.1; 384]; // Your embedding vector
//! let id = engine.add(
//!     "Project deadline moved to March 15th".to_string(),
//!     embedding,
//!     None,
//!     None,
//!     None,
//!     None,
//! ).unwrap();
//!
//! // Retrieve a memory
//! let memory = engine.get(&id).unwrap();
//!
//! // Close the database
//! engine.close().unwrap();
//! ```
//!
//! # Architecture
//!
//! MnemeFusion is built on:
//! - **redb**: ACID-compliant embedded storage
//! - **usearch**: High-performance vector similarity search (HNSW)
//! - **petgraph**: Graph algorithms for causal and entity relationships
//!
//! All data is stored in a single `.mfdb` file, making it easy to backup,
//! version, and deploy.

pub mod config;
pub mod error;
pub mod graph;
pub mod index;
pub mod ingest;
pub mod memory;
pub mod query;
pub mod slm;
pub mod storage;
pub mod types;
pub mod util;

// Native LLM inference module (only available with entity-extraction feature)
#[cfg(feature = "entity-extraction")]
pub mod inference;

// Entity extraction module (only available with entity-extraction feature)
#[cfg(feature = "entity-extraction")]
pub mod extraction;

// Embedding engine module (only available with embedding-onnx feature)
pub mod embedding;

// Public API exports
pub use config::Config;
pub use error::{Error, Result};
pub use graph::{CausalEdge, CausalPath, CausalTraversalResult, EntityQueryResult, GraphManager};
pub use index::{TemporalIndex, TemporalResult, VectorIndex, VectorIndexConfig, VectorResult};
pub use ingest::{EntityExtractor, SimpleEntityExtractor};
pub use memory::{contextualize_for_embedding, first_person_to_third, EmbeddingFn, MemoryEngine, ScopedMemory};
pub use query::{
    AdaptiveWeightConfig, FusedResult, FusionEngine, IntentClassification, IntentClassifier,
    IntentWeights, QueryIntent, QueryPlanner,
};

// SLM exports (only available when slm feature is enabled)
#[cfg(feature = "slm")]
pub use slm::{SlmClassifier, SlmConfig};

#[cfg(not(feature = "slm"))]
pub use slm::SlmConfig; // Config always available, but SlmClassifier only with feature

// Native inference exports (only available with entity-extraction feature)
#[cfg(feature = "entity-extraction")]
pub use inference::{InferenceEngine, JsonGrammar};

// Entity extraction exports (only available with entity-extraction feature)
#[cfg(feature = "entity-extraction")]
pub use extraction::{ExtractedEntity, ExtractedFact, ExtractionResult, LlmEntityExtractor, ModelTier};

// Embedding engine export (only available with embedding-onnx feature)
#[cfg(feature = "embedding-onnx")]
pub use embedding::EmbeddingEngine;

pub use types::{
    AddResult, BatchError, BatchResult, Entity, EntityId, FilterOp, Memory, MemoryId, MemoryInput,
    MetadataFilter, Source, SourceType, Timestamp, UpsertResult, NAMESPACE_METADATA_KEY,
    SOURCE_METADATA_KEY,
};
