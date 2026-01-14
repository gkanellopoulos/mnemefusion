//! Storage layer for MnemeFusion
//!
//! This module handles persistent storage using redb.

pub mod engine;
pub mod format;

pub use engine::StorageEngine;
pub use format::FileHeader;
