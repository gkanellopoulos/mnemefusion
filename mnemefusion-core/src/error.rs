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

    /// Invalid source
    #[error("Invalid source: {0}")]
    InvalidSource(String),

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

    /// Namespace mismatch
    #[error("Namespace mismatch: expected '{expected}', found '{found}'")]
    NamespaceMismatch { expected: String, found: String },

    /// Database corruption detected
    #[error("Database corruption detected: {0}")]
    DatabaseCorruption(String),

    /// File truncated or incomplete
    #[error("File truncated or incomplete: {0}")]
    FileTruncated(String),

    /// SLM feature not available (compiled without 'slm' feature)
    #[error("SLM feature not available - compile with 'slm' feature to enable")]
    SlmNotAvailable,

    /// SLM initialization error
    #[error("SLM initialization error: {0}")]
    SlmInitialization(String),

    /// SLM inference error
    #[error("SLM inference error: {0}")]
    SlmInference(String),

    /// SLM timeout error
    #[error("SLM inference timeout after {0}ms")]
    SlmTimeout(u64),
}

/// Result type alias for MnemeFusion operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Check if this error is recoverable
    ///
    /// Recoverable errors can typically be retried or worked around.
    /// Non-recoverable errors indicate serious problems that require user intervention.
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Recoverable errors
            Error::MemoryNotFound(_) => true,
            Error::EntityNotFound(_) => true,
            Error::NamespaceMismatch { .. } => true,
            Error::InvalidParameter(_) => true,
            Error::InvalidEmbeddingDimension { .. } => true,

            // Non-recoverable errors
            Error::DatabaseCorruption(_) => false,
            Error::FileTruncated(_) => false,
            Error::InvalidFormat(_) => false,
            Error::UnsupportedVersion(..) => false,

            // Storage/IO errors might be recoverable
            Error::Storage(_) => true,
            Error::Io(_) => true,

            // Other errors are situational
            _ => false,
        }
    }

    /// Get a user-friendly error message with troubleshooting hints
    pub fn user_message(&self) -> String {
        match self {
            Error::InvalidEmbeddingDimension { expected, got } => {
                format!(
                    "Embedding dimension mismatch: expected {} dimensions, but got {}.\n\
                     Hint: Ensure all embeddings use the same model and dimension size.\n\
                     You can set the dimension in Config with .with_embedding_dim({})",
                    expected, got, expected
                )
            }
            Error::DatabaseCorruption(msg) => {
                format!(
                    "Database corruption detected: {}\n\
                     The database file may be corrupted or incomplete.\n\
                     Hint: Try restoring from a backup if available, or create a new database.",
                    msg
                )
            }
            Error::FileTruncated(msg) => {
                format!(
                    "Database file is truncated: {}\n\
                     The file may have been corrupted during a previous operation.\n\
                     Hint: Restore from backup or delete the file to start fresh.",
                    msg
                )
            }
            Error::UnsupportedVersion(found, current) => {
                if found > current {
                    format!(
                        "Database version {} is newer than supported version {}.\n\
                         Hint: Update MnemeFusion to the latest version.",
                        found, current
                    )
                } else {
                    format!(
                        "Database version {} is older than current version {}.\n\
                         Migration may be required.",
                        found, current
                    )
                }
            }
            Error::NamespaceMismatch { expected, found } => {
                format!(
                    "Namespace mismatch: operation expected '{}' but memory is in '{}'.\n\
                     Hint: Verify you're using the correct namespace for this operation.",
                    expected, found
                )
            }
            Error::VectorIndex(msg) => {
                format!(
                    "Vector index error: {}\n\
                     Hint: This may indicate corrupted index data. Try reopening the database.",
                    msg
                )
            }
            _ => self.to_string(),
        }
    }

    /// Check if this is a corruption-related error
    pub fn is_corruption(&self) -> bool {
        matches!(
            self,
            Error::DatabaseCorruption(_) | Error::FileTruncated(_) | Error::InvalidFormat(_)
        )
    }

    /// Check if this is a version-related error
    pub fn is_version_error(&self) -> bool {
        matches!(self, Error::UnsupportedVersion(..))
    }
}

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

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        if err.is_data() {
            Error::Deserialization(err.to_string())
        } else {
            Error::Serialization(err.to_string())
        }
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

    #[test]
    fn test_is_recoverable() {
        // Recoverable errors
        assert!(Error::MemoryNotFound("id".to_string()).is_recoverable());
        assert!(Error::EntityNotFound("id".to_string()).is_recoverable());
        assert!(Error::InvalidParameter("test".to_string()).is_recoverable());
        assert!(Error::InvalidEmbeddingDimension {
            expected: 384,
            got: 512
        }
        .is_recoverable());

        // Non-recoverable errors
        assert!(!Error::DatabaseCorruption("test".to_string()).is_recoverable());
        assert!(!Error::FileTruncated("test".to_string()).is_recoverable());
        assert!(!Error::InvalidFormat("test").is_recoverable());
        assert!(!Error::UnsupportedVersion(2, 1).is_recoverable());
    }

    #[test]
    fn test_is_corruption() {
        assert!(Error::DatabaseCorruption("test".to_string()).is_corruption());
        assert!(Error::FileTruncated("test".to_string()).is_corruption());
        assert!(Error::InvalidFormat("test").is_corruption());

        assert!(!Error::MemoryNotFound("id".to_string()).is_corruption());
        assert!(!Error::InvalidParameter("test".to_string()).is_corruption());
    }

    #[test]
    fn test_is_version_error() {
        assert!(Error::UnsupportedVersion(2, 1).is_version_error());
        assert!(!Error::DatabaseCorruption("test".to_string()).is_version_error());
    }

    #[test]
    fn test_user_message() {
        // Test dimension error message includes hint
        let err = Error::InvalidEmbeddingDimension {
            expected: 384,
            got: 512,
        };
        let msg = err.user_message();
        assert!(msg.contains("expected 384"));
        assert!(msg.contains("got 512"));
        assert!(msg.contains("Hint"));

        // Test corruption error includes recovery suggestion
        let err = Error::DatabaseCorruption("bad data".to_string());
        let msg = err.user_message();
        assert!(msg.contains("corruption"));
        assert!(msg.contains("backup"));

        // Test version error distinguishes newer vs older
        let err = Error::UnsupportedVersion(5, 1);
        let msg = err.user_message();
        assert!(msg.contains("newer"));
        assert!(msg.contains("Update"));
    }

    #[test]
    fn test_unsupported_version_messages() {
        // Newer version
        let err = Error::UnsupportedVersion(5, 1);
        let msg = err.user_message();
        assert!(msg.contains("newer"));

        // Older version (edge case, but handled)
        let err = Error::UnsupportedVersion(1, 5);
        let msg = err.user_message();
        assert!(msg.contains("older") || msg.contains("Migration"));
    }
}
