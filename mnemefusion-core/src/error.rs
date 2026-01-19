//! Error types for MnemeFusion
//!
//! This module defines all error types that can occur during MnemeFusion operations.
//! Uses thiserror for ergonomic error handling.

/// Main error type for MnemeFusion operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Storage engine error (redb)
    #[error("Storage error: {0}")]
    Storage(#[from] redb::Error),

    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// Table error
    #[error("Table error: {0}")]
    Table(String),

    /// Storage transaction error
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// Storage commit error
    #[error("Commit error: {0}")]
    Commit(String),

    /// Vector index error
    #[error("Vector index error: {0}")]
    VectorIndex(String),

    /// Invalid file format
    #[error("Invalid file format: {0}")]
    InvalidFormat(&'static str),

    /// Unsupported version
    #[error("Unsupported version: {0} (current version: {1})")]
    UnsupportedVersion(u32, u32),

    /// Memory not found
    #[error("Memory not found: {0}")]
    MemoryNotFound(String),

    /// Entity not found
    #[error("Entity not found: {0}")]
    EntityNotFound(String),

    /// Invalid embedding dimension
    #[error("Invalid embedding dimension: expected {expected}, got {got}")]
    InvalidEmbeddingDimension { expected: usize, got: usize },

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid memory ID format
    #[error("Invalid memory ID: {0}")]
    InvalidMemoryId(String),

    /// Invalid timestamp
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Database already exists
    #[error("Database already exists at path: {0}")]
    DatabaseExists(String),

    /// Database not found
    #[error("Database not found at path: {0}")]
    DatabaseNotFound(String),
}

/// Result type alias for MnemeFusion operations
pub type Result<T> = std::result::Result<T, Error>;

// Convert redb specific errors to our Error type
impl From<redb::DatabaseError> for Error {
    fn from(err: redb::DatabaseError) -> Self {
        Error::Database(err.to_string())
    }
}

impl From<redb::TransactionError> for Error {
    fn from(err: redb::TransactionError) -> Self {
        Error::Transaction(err.to_string())
    }
}

impl From<redb::TableError> for Error {
    fn from(err: redb::TableError) -> Self {
        Error::Table(err.to_string())
    }
}

impl From<redb::CommitError> for Error {
    fn from(err: redb::CommitError) -> Self {
        Error::Commit(err.to_string())
    }
}

impl From<redb::StorageError> for Error {
    fn from(err: redb::StorageError) -> Self {
        Error::Storage(redb::Error::from(err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::InvalidFormat("bad magic number");
        assert_eq!(err.to_string(), "Invalid file format: bad magic number");

        let err = Error::InvalidEmbeddingDimension {
            expected: 384,
            got: 512,
        };
        assert_eq!(
            err.to_string(),
            "Invalid embedding dimension: expected 384, got 512"
        );
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }
}
