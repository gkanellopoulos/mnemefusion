//! Core types for MnemeFusion
//!
//! This module contains the fundamental data types used throughout the system.

pub mod batch;
pub mod dedup;
pub mod entity;
pub mod memory;
pub mod source;
pub mod timestamp;

pub use batch::{BatchError, BatchResult, MemoryInput};
pub use dedup::{AddResult, UpsertResult};
pub use entity::{Entity, EntityId};
pub use memory::{Memory, MemoryId};
pub use source::{Source, SourceType, SOURCE_METADATA_KEY};
pub use timestamp::Timestamp;
