//! Content hashing utilities for deduplication
//!
//! Uses SHA-256 for content hashing to detect duplicate memories.

use sha2::{Digest, Sha256};

/// Compute SHA-256 hash of content string
///
/// Returns hex-encoded hash string for use as storage key.
///
/// # Examples
///
/// ```
/// use mnemefusion_core::util::hash::hash_content;
///
/// let hash1 = hash_content("test content");
/// let hash2 = hash_content("test content");
/// let hash3 = hash_content("different content");
///
/// assert_eq!(hash1, hash2);
/// assert_ne!(hash1, hash3);
/// ```
pub fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

/// Compute SHA-256 hash with normalization
///
/// Normalizes whitespace before hashing to catch duplicates with minor formatting differences.
/// Converts multiple spaces to single space and trims.
///
/// # Examples
///
/// ```
/// use mnemefusion_core::util::hash::hash_content_normalized;
///
/// let hash1 = hash_content_normalized("test  content");  // double space
/// let hash2 = hash_content_normalized("test content");   // single space
///
/// assert_eq!(hash1, hash2); // Treated as same content
/// ```
pub fn hash_content_normalized(content: &str) -> String {
    // Normalize: trim, collapse multiple spaces to single
    let normalized: String = content.split_whitespace().collect::<Vec<&str>>().join(" ");
    hash_content(&normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_content_deterministic() {
        let content = "test content";
        let hash1 = hash_content(content);
        let hash2 = hash_content(content);

        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 produces 64 hex chars
    }

    #[test]
    fn test_hash_content_different() {
        let hash1 = hash_content("content 1");
        let hash2 = hash_content("content 2");

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_content_empty() {
        let hash = hash_content("");
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_hash_content_case_sensitive() {
        let hash1 = hash_content("Test");
        let hash2 = hash_content("test");

        assert_ne!(hash1, hash2); // Hashing is case-sensitive
    }

    #[test]
    fn test_hash_content_normalized() {
        let hash1 = hash_content_normalized("test  content");
        let hash2 = hash_content_normalized("test content");
        let hash3 = hash_content_normalized("  test   content  ");

        assert_eq!(hash1, hash2);
        assert_eq!(hash1, hash3);
    }

    #[test]
    fn test_hash_normalized_vs_regular() {
        let content = "test content";
        let hash_regular = hash_content(content);
        let hash_normalized = hash_content_normalized(content);

        // For already-normalized content, they should match
        assert_eq!(hash_regular, hash_normalized);
    }

    #[test]
    fn test_hash_normalized_different_whitespace() {
        let hash1 = hash_content_normalized("hello world");
        let hash2 = hash_content_normalized("hello\nworld");
        let hash3 = hash_content_normalized("hello\t\tworld");

        // All whitespace collapsed to single space
        assert_eq!(hash1, hash2);
        assert_eq!(hash1, hash3);
    }
}
