//! Core types for MnemeFusion
//!
//! This module contains the fundamental data types used throughout the system.

pub mod memory;
pub mod timestamp;

pub use memory::{Memory, MemoryId};
pub use timestamp::Timestamp;
