//! Entity types for entity graph functionality
//!
//! Entities represent people, organizations, locations, or other named concepts
//! that appear in memories. The entity graph tracks relationships between
//! memories and entities.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for an entity
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(Uuid);

impl EntityId {
    /// Create a new random entity ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Parse an entity ID from a string
    pub fn parse(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s)
            .map_err(|e| crate::Error::InvalidMemoryId(format!("Invalid entity ID: {}", e)))?;
        Ok(Self(uuid))
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Convert to bytes for storage
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let uuid = Uuid::from_slice(bytes)
            .map_err(|e| crate::Error::InvalidMemoryId(format!("Invalid entity ID bytes: {}", e)))?;
        Ok(Self(uuid))
    }
}

impl Default for EntityId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An entity extracted from or linked to memories
///
/// Entities represent named concepts like people, organizations, projects, etc.
/// that appear across multiple memories. The entity graph tracks which memories
/// mention which entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier for this entity
    pub id: EntityId,

    /// Canonical name of the entity
    /// Stored in original case, but lookups are case-insensitive
    pub name: String,

    /// Optional metadata about the entity
    /// Can be used to store entity type, aliases, descriptions, etc.
    pub metadata: HashMap<String, String>,

    /// Number of memories that mention this entity
    /// Updated when memories are added/deleted
    pub mention_count: usize,
}

impl Entity {
    /// Create a new entity with the given name
    ///
    /// # Arguments
    ///
    /// * `name` - The canonical name for this entity
    ///
    /// # Example
    ///
    /// ```
    /// use mnemefusion_core::Entity;
    ///
    /// let entity = Entity::new("Project Alpha");
    /// assert_eq!(entity.name, "Project Alpha");
    /// assert_eq!(entity.mention_count, 0);
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: EntityId::new(),
            name: name.into(),
            metadata: HashMap::new(),
            mention_count: 0,
        }
    }

    /// Create an entity with custom ID (for loading from storage)
    pub fn with_id(id: EntityId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            metadata: HashMap::new(),
            mention_count: 0,
        }
    }

    /// Add metadata to this entity
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get the normalized (lowercase) name for lookups
    pub fn normalized_name(&self) -> String {
        self.name.to_lowercase()
    }

    /// Increment the mention count
    pub fn increment_mentions(&mut self) {
        self.mention_count += 1;
    }

    /// Decrement the mention count
    pub fn decrement_mentions(&mut self) {
        self.mention_count = self.mention_count.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_id_new() {
        let id1 = EntityId::new();
        let id2 = EntityId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_entity_id_parse() {
        let id = EntityId::new();
        let s = id.to_string();
        let parsed = EntityId::parse(&s).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_entity_id_bytes() {
        let id = EntityId::new();
        let bytes = id.as_bytes();
        let restored = EntityId::from_bytes(bytes).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_entity_new() {
        let entity = Entity::new("Project Alpha");
        assert_eq!(entity.name, "Project Alpha");
        assert_eq!(entity.mention_count, 0);
        assert!(entity.metadata.is_empty());
    }

    #[test]
    fn test_entity_normalized_name() {
        let entity = Entity::new("Project Alpha");
        assert_eq!(entity.normalized_name(), "project alpha");

        let entity2 = Entity::new("PROJECT ALPHA");
        assert_eq!(entity2.normalized_name(), "project alpha");
    }

    #[test]
    fn test_entity_with_metadata() {
        let entity = Entity::new("John Smith")
            .with_metadata("type", "person")
            .with_metadata("role", "engineer");

        assert_eq!(entity.metadata.get("type"), Some(&"person".to_string()));
        assert_eq!(entity.metadata.get("role"), Some(&"engineer".to_string()));
    }

    #[test]
    fn test_entity_mention_count() {
        let mut entity = Entity::new("Test");
        assert_eq!(entity.mention_count, 0);

        entity.increment_mentions();
        assert_eq!(entity.mention_count, 1);

        entity.increment_mentions();
        assert_eq!(entity.mention_count, 2);

        entity.decrement_mentions();
        assert_eq!(entity.mention_count, 1);

        entity.decrement_mentions();
        assert_eq!(entity.mention_count, 0);

        // Should not go negative
        entity.decrement_mentions();
        assert_eq!(entity.mention_count, 0);
    }

    #[test]
    fn test_entity_serialization() {
        let entity = Entity::new("Test Entity")
            .with_metadata("key", "value");

        let json = serde_json::to_string(&entity).unwrap();
        let deserialized: Entity = serde_json::from_str(&json).unwrap();

        assert_eq!(entity.name, deserialized.name);
        assert_eq!(entity.metadata, deserialized.metadata);
    }
}
