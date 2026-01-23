//! Deduplication and upsert result types
//!
//! This module provides types for tracking deduplication and upsert operations.

use super::MemoryId;

/// Result of an add operation with deduplication
///
/// Provides information about whether a memory was newly created
/// or if a duplicate was found.
///
/// # Examples
///
/// ```
/// use mnemefusion_core::types::dedup::AddResult;
/// use mnemefusion_core::types::MemoryId;
///
/// // New memory created
/// let result = AddResult::created(MemoryId::new());
/// assert!(result.created);
/// assert!(result.existing_id.is_none());
///
/// // Duplicate found
/// let existing = MemoryId::new();
/// let result = AddResult::duplicate(existing.clone());
/// assert!(!result.created);
/// assert_eq!(result.existing_id, Some(existing));
/// ```
#[derive(Debug, Clone)]
pub struct AddResult {
    /// ID of the memory (either newly created or existing)
    pub id: MemoryId,

    /// True if a new memory was created, false if duplicate was found
    pub created: bool,

    /// If duplicate was found, this contains the existing memory ID
    /// Same as `id` when `created` is false
    pub existing_id: Option<MemoryId>,
}

impl AddResult {
    /// Create AddResult for a newly created memory
    pub fn created(id: MemoryId) -> Self {
        Self {
            id,
            created: true,
            existing_id: None,
        }
    }

    /// Create AddResult for a duplicate memory
    pub fn duplicate(existing_id: MemoryId) -> Self {
        Self {
            id: existing_id.clone(),
            created: false,
            existing_id: Some(existing_id),
        }
    }

    /// Check if this was a newly created memory
    pub fn is_created(&self) -> bool {
        self.created
    }

    /// Check if this was a duplicate
    pub fn is_duplicate(&self) -> bool {
        !self.created
    }
}

/// Result of an upsert operation
///
/// Provides detailed information about whether a memory was created or updated,
/// including the previous content if updated.
///
/// # Examples
///
/// ```
/// use mnemefusion_core::types::dedup::UpsertResult;
/// use mnemefusion_core::types::MemoryId;
///
/// // New memory created
/// let result = UpsertResult::created(MemoryId::new());
/// assert!(result.created);
/// assert!(!result.updated);
///
/// // Existing memory updated
/// let result = UpsertResult::updated(MemoryId::new(), Some("old content".to_string()));
/// assert!(!result.created);
/// assert!(result.updated);
/// assert_eq!(result.previous_content, Some("old content".to_string()));
/// ```
#[derive(Debug, Clone)]
pub struct UpsertResult {
    /// ID of the memory (either newly created or existing)
    pub id: MemoryId,

    /// True if a new memory was created
    pub created: bool,

    /// True if an existing memory was updated
    pub updated: bool,

    /// Previous content if memory was updated (None if created)
    pub previous_content: Option<String>,
}

impl UpsertResult {
    /// Create UpsertResult for a newly created memory
    pub fn created(id: MemoryId) -> Self {
        Self {
            id,
            created: true,
            updated: false,
            previous_content: None,
        }
    }

    /// Create UpsertResult for an updated memory
    pub fn updated(id: MemoryId, previous_content: Option<String>) -> Self {
        Self {
            id,
            created: false,
            updated: true,
            previous_content,
        }
    }

    /// Check if this was a newly created memory
    pub fn is_created(&self) -> bool {
        self.created
    }

    /// Check if this was an update
    pub fn is_updated(&self) -> bool {
        self.updated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_result_created() {
        let id = MemoryId::new();
        let result = AddResult::created(id.clone());

        assert_eq!(result.id, id);
        assert!(result.created);
        assert!(result.is_created());
        assert!(!result.is_duplicate());
        assert!(result.existing_id.is_none());
    }

    #[test]
    fn test_add_result_duplicate() {
        let existing = MemoryId::new();
        let result = AddResult::duplicate(existing.clone());

        assert_eq!(result.id, existing);
        assert!(!result.created);
        assert!(!result.is_created());
        assert!(result.is_duplicate());
        assert_eq!(result.existing_id, Some(existing));
    }

    #[test]
    fn test_upsert_result_created() {
        let id = MemoryId::new();
        let result = UpsertResult::created(id.clone());

        assert_eq!(result.id, id);
        assert!(result.created);
        assert!(result.is_created());
        assert!(!result.updated);
        assert!(!result.is_updated());
        assert!(result.previous_content.is_none());
    }

    #[test]
    fn test_upsert_result_updated() {
        let id = MemoryId::new();
        let prev = "old content".to_string();
        let result = UpsertResult::updated(id.clone(), Some(prev.clone()));

        assert_eq!(result.id, id);
        assert!(!result.created);
        assert!(!result.is_created());
        assert!(result.updated);
        assert!(result.is_updated());
        assert_eq!(result.previous_content, Some(prev));
    }

    #[test]
    fn test_upsert_result_updated_no_previous() {
        let id = MemoryId::new();
        let result = UpsertResult::updated(id.clone(), None);

        assert!(result.updated);
        assert!(result.previous_content.is_none());
    }
}
