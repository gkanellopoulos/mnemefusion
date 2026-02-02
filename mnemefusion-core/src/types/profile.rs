//! Entity Profile types for aggregated entity facts
//!
//! Entity Profiles aggregate facts about entities across all memories.
//! When a query asks "What is Caroline researching?", we can directly look up
//! Caroline's profile to find `research: "adoption agencies"` instead of
//! relying on semantic similarity alone.
//!
//! # Architecture
//!
//! Facts are extracted from memories at ingestion time using the SLM.
//! Each fact includes:
//! - The entity it's about
//! - The type of fact (e.g., "research_topic", "occupation")
//! - The fact value
//! - Confidence score
//! - Source memory ID for provenance tracking
//!
//! # Example
//!
//! ```ignore
//! // Memory: "Caroline found an adoption agency to research her biological parents"
//! // Extracts:
//! //   - Entity: Caroline
//! //   - Fact: { type: "research_topic", value: "adoption agencies", confidence: 0.9 }
//! //   - Fact: { type: "goal", value: "find biological parents", confidence: 0.85 }
//! ```

use crate::types::{EntityId, MemoryId, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An aggregated profile for an entity across all memories
///
/// Profiles accumulate facts about entities from multiple memories,
/// enabling direct fact lookup instead of relying on semantic similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityProfile {
    /// The entity this profile describes
    pub entity_id: EntityId,

    /// Canonical name of the entity
    pub name: String,

    /// Entity type: "person", "organization", "location", "concept"
    pub entity_type: String,

    /// Aggregated facts about this entity
    /// Key: fact_type (e.g., "research_topic", "occupation", "location")
    /// Value: list of facts with that type
    pub facts: HashMap<String, Vec<EntityFact>>,

    /// Memory IDs that mention this entity
    pub source_memories: Vec<MemoryId>,

    /// Last updated timestamp
    pub updated_at: Timestamp,
}

impl EntityProfile {
    /// Create a new empty profile for an entity
    ///
    /// # Arguments
    ///
    /// * `entity_id` - The entity ID this profile describes
    /// * `name` - Canonical name of the entity
    /// * `entity_type` - Type of entity (person, organization, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use mnemefusion_core::types::{EntityId, EntityProfile};
    ///
    /// let profile = EntityProfile::new(
    ///     EntityId::new(),
    ///     "Caroline".to_string(),
    ///     "person".to_string(),
    /// );
    /// assert!(profile.facts.is_empty());
    /// ```
    pub fn new(entity_id: EntityId, name: String, entity_type: String) -> Self {
        Self {
            entity_id,
            name,
            entity_type,
            facts: HashMap::new(),
            source_memories: Vec::new(),
            updated_at: Timestamp::now(),
        }
    }

    /// Add a fact to this profile
    ///
    /// Facts are organized by type. Multiple facts of the same type can exist
    /// (e.g., multiple occupations over time).
    ///
    /// # Arguments
    ///
    /// * `fact` - The fact to add
    ///
    /// # Example
    ///
    /// ```
    /// use mnemefusion_core::types::{EntityId, EntityProfile, EntityFact, MemoryId, Timestamp};
    ///
    /// let mut profile = EntityProfile::new(
    ///     EntityId::new(),
    ///     "Caroline".to_string(),
    ///     "person".to_string(),
    /// );
    ///
    /// profile.add_fact(EntityFact {
    ///     fact_type: "research_topic".to_string(),
    ///     value: "adoption agencies".to_string(),
    ///     confidence: 0.9,
    ///     source_memory: MemoryId::new(),
    ///     extracted_at: Timestamp::now(),
    /// });
    ///
    /// assert_eq!(profile.facts.len(), 1);
    /// ```
    pub fn add_fact(&mut self, fact: EntityFact) {
        self.facts
            .entry(fact.fact_type.clone())
            .or_default()
            .push(fact);
        self.updated_at = Timestamp::now();
    }

    /// Add a source memory to this profile
    ///
    /// Tracks which memories have contributed facts to this profile.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory ID to add
    pub fn add_source_memory(&mut self, memory_id: MemoryId) {
        if !self.source_memories.contains(&memory_id) {
            self.source_memories.push(memory_id);
            self.updated_at = Timestamp::now();
        }
    }

    /// Remove all facts from a specific source memory
    ///
    /// Used when a memory is deleted to clean up stale facts.
    /// Also removes the memory from source_memories.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory ID whose facts should be removed
    ///
    /// # Example
    ///
    /// ```
    /// use mnemefusion_core::types::{EntityId, EntityProfile, EntityFact, MemoryId, Timestamp};
    ///
    /// let mut profile = EntityProfile::new(
    ///     EntityId::new(),
    ///     "Test".to_string(),
    ///     "person".to_string(),
    /// );
    ///
    /// let memory_id = MemoryId::new();
    /// profile.add_fact(EntityFact {
    ///     fact_type: "occupation".to_string(),
    ///     value: "engineer".to_string(),
    ///     confidence: 0.9,
    ///     source_memory: memory_id.clone(),
    ///     extracted_at: Timestamp::now(),
    /// });
    /// profile.add_source_memory(memory_id.clone());
    ///
    /// profile.remove_facts_from_memory(&memory_id);
    ///
    /// assert!(profile.facts.get("occupation").unwrap().is_empty());
    /// assert!(!profile.source_memories.contains(&memory_id));
    /// ```
    pub fn remove_facts_from_memory(&mut self, memory_id: &MemoryId) {
        for facts in self.facts.values_mut() {
            facts.retain(|f| &f.source_memory != memory_id);
        }
        self.source_memories.retain(|id| id != memory_id);
        self.updated_at = Timestamp::now();
    }

    /// Get facts of a specific type, sorted by confidence (highest first)
    ///
    /// # Arguments
    ///
    /// * `fact_type` - The type of facts to retrieve
    ///
    /// # Returns
    ///
    /// Vector of references to facts of the specified type, sorted by confidence
    ///
    /// # Example
    ///
    /// ```
    /// use mnemefusion_core::types::{EntityId, EntityProfile, EntityFact, MemoryId, Timestamp};
    ///
    /// let mut profile = EntityProfile::new(
    ///     EntityId::new(),
    ///     "Alice".to_string(),
    ///     "person".to_string(),
    /// );
    ///
    /// profile.add_fact(EntityFact {
    ///     fact_type: "occupation".to_string(),
    ///     value: "engineer".to_string(),
    ///     confidence: 0.7,
    ///     source_memory: MemoryId::new(),
    ///     extracted_at: Timestamp::now(),
    /// });
    ///
    /// profile.add_fact(EntityFact {
    ///     fact_type: "occupation".to_string(),
    ///     value: "architect".to_string(),
    ///     confidence: 0.9,
    ///     source_memory: MemoryId::new(),
    ///     extracted_at: Timestamp::now(),
    /// });
    ///
    /// let occupations = profile.get_facts("occupation");
    /// assert_eq!(occupations.len(), 2);
    /// assert_eq!(occupations[0].value, "architect"); // Higher confidence first
    /// assert_eq!(occupations[1].value, "engineer");
    /// ```
    pub fn get_facts(&self, fact_type: &str) -> Vec<&EntityFact> {
        let mut facts: Vec<_> = self
            .facts
            .get(fact_type)
            .map(|v| v.iter().collect())
            .unwrap_or_default();
        facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts
    }

    /// Get all fact types present in this profile
    ///
    /// # Returns
    ///
    /// Vector of fact type strings
    pub fn fact_types(&self) -> Vec<&str> {
        self.facts.keys().map(|s| s.as_str()).collect()
    }

    /// Get the total number of facts across all types
    pub fn total_facts(&self) -> usize {
        self.facts.values().map(|v| v.len()).sum()
    }

    /// Check if the profile has any facts
    pub fn is_empty(&self) -> bool {
        self.facts.values().all(|v| v.is_empty())
    }

    /// Get the highest confidence fact of a specific type
    ///
    /// # Arguments
    ///
    /// * `fact_type` - The type of fact to retrieve
    ///
    /// # Returns
    ///
    /// The fact with highest confidence, or None if no facts of that type
    pub fn get_best_fact(&self, fact_type: &str) -> Option<&EntityFact> {
        self.get_facts(fact_type).first().copied()
    }
}

/// A single fact extracted about an entity
///
/// Facts are the atomic units of knowledge about entities.
/// They include provenance information (source memory) for tracing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EntityFact {
    /// The type of fact: "research_topic", "occupation", "relationship", "goal", etc.
    pub fact_type: String,

    /// The fact value: "adoption agencies", "software engineer", etc.
    pub value: String,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,

    /// Source memory this fact was extracted from
    pub source_memory: MemoryId,

    /// When this fact was extracted
    pub extracted_at: Timestamp,
}

impl EntityFact {
    /// Create a new entity fact
    ///
    /// # Arguments
    ///
    /// * `fact_type` - The type of fact
    /// * `value` - The fact value
    /// * `confidence` - Confidence score (clamped to 0.0-1.0)
    /// * `source_memory` - The memory this fact was extracted from
    ///
    /// # Example
    ///
    /// ```
    /// use mnemefusion_core::types::{EntityFact, MemoryId};
    ///
    /// let fact = EntityFact::new(
    ///     "occupation",
    ///     "software engineer",
    ///     0.95,
    ///     MemoryId::new(),
    /// );
    /// assert_eq!(fact.fact_type, "occupation");
    /// assert_eq!(fact.confidence, 0.95);
    /// ```
    pub fn new(
        fact_type: impl Into<String>,
        value: impl Into<String>,
        confidence: f32,
        source_memory: MemoryId,
    ) -> Self {
        Self {
            fact_type: fact_type.into(),
            value: value.into(),
            confidence: confidence.clamp(0.0, 1.0),
            source_memory,
            extracted_at: Timestamp::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_profile_new() {
        let entity_id = EntityId::new();
        let profile = EntityProfile::new(entity_id.clone(), "Alice".to_string(), "person".to_string());

        assert_eq!(profile.entity_id, entity_id);
        assert_eq!(profile.name, "Alice");
        assert_eq!(profile.entity_type, "person");
        assert!(profile.facts.is_empty());
        assert!(profile.source_memories.is_empty());
    }

    #[test]
    fn test_entity_profile_add_fact() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        let memory_id = MemoryId::new();
        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            memory_id.clone(),
        ));

        assert_eq!(profile.facts.len(), 1);
        assert_eq!(profile.facts.get("occupation").unwrap().len(), 1);
        assert_eq!(
            profile.facts.get("occupation").unwrap()[0].value,
            "engineer"
        );
    }

    #[test]
    fn test_entity_profile_add_multiple_facts_same_type() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.7,
            MemoryId::new(),
        ));
        profile.add_fact(EntityFact::new(
            "occupation",
            "architect",
            0.9,
            MemoryId::new(),
        ));

        assert_eq!(profile.facts.get("occupation").unwrap().len(), 2);
    }

    #[test]
    fn test_entity_profile_get_facts_sorted() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        profile.add_fact(EntityFact::new(
            "skill",
            "Rust",
            0.7,
            MemoryId::new(),
        ));
        profile.add_fact(EntityFact::new(
            "skill",
            "Python",
            0.95,
            MemoryId::new(),
        ));
        profile.add_fact(EntityFact::new(
            "skill",
            "Go",
            0.8,
            MemoryId::new(),
        ));

        let skills = profile.get_facts("skill");
        assert_eq!(skills.len(), 3);
        assert_eq!(skills[0].value, "Python"); // 0.95
        assert_eq!(skills[1].value, "Go"); // 0.8
        assert_eq!(skills[2].value, "Rust"); // 0.7
    }

    #[test]
    fn test_entity_profile_get_facts_empty() {
        let profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        let facts = profile.get_facts("nonexistent");
        assert!(facts.is_empty());
    }

    #[test]
    fn test_entity_profile_remove_facts_from_memory() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        let memory1 = MemoryId::new();
        let memory2 = MemoryId::new();

        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            memory1.clone(),
        ));
        profile.add_fact(EntityFact::new(
            "skill",
            "Rust",
            0.8,
            memory1.clone(),
        ));
        profile.add_fact(EntityFact::new(
            "occupation",
            "architect",
            0.7,
            memory2.clone(),
        ));
        profile.add_source_memory(memory1.clone());
        profile.add_source_memory(memory2.clone());

        // Remove facts from memory1
        profile.remove_facts_from_memory(&memory1);

        // Only architect should remain
        assert_eq!(profile.facts.get("occupation").unwrap().len(), 1);
        assert_eq!(profile.facts.get("occupation").unwrap()[0].value, "architect");

        // Skill should be empty
        assert!(profile.facts.get("skill").unwrap().is_empty());

        // Source memories should not include memory1
        assert!(!profile.source_memories.contains(&memory1));
        assert!(profile.source_memories.contains(&memory2));
    }

    #[test]
    fn test_entity_profile_add_source_memory() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        let memory1 = MemoryId::new();
        let memory2 = MemoryId::new();

        profile.add_source_memory(memory1.clone());
        profile.add_source_memory(memory2.clone());
        profile.add_source_memory(memory1.clone()); // Duplicate

        assert_eq!(profile.source_memories.len(), 2);
    }

    #[test]
    fn test_entity_profile_fact_types() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            MemoryId::new(),
        ));
        profile.add_fact(EntityFact::new(
            "skill",
            "Rust",
            0.8,
            MemoryId::new(),
        ));
        profile.add_fact(EntityFact::new(
            "location",
            "Seattle",
            0.7,
            MemoryId::new(),
        ));

        let types = profile.fact_types();
        assert_eq!(types.len(), 3);
        assert!(types.contains(&"occupation"));
        assert!(types.contains(&"skill"));
        assert!(types.contains(&"location"));
    }

    #[test]
    fn test_entity_profile_total_facts() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        assert_eq!(profile.total_facts(), 0);

        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            MemoryId::new(),
        ));
        assert_eq!(profile.total_facts(), 1);

        profile.add_fact(EntityFact::new(
            "skill",
            "Rust",
            0.8,
            MemoryId::new(),
        ));
        profile.add_fact(EntityFact::new(
            "skill",
            "Python",
            0.7,
            MemoryId::new(),
        ));
        assert_eq!(profile.total_facts(), 3);
    }

    #[test]
    fn test_entity_profile_is_empty() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        assert!(profile.is_empty());

        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            MemoryId::new(),
        ));
        assert!(!profile.is_empty());
    }

    #[test]
    fn test_entity_profile_get_best_fact() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        assert!(profile.get_best_fact("occupation").is_none());

        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.7,
            MemoryId::new(),
        ));
        profile.add_fact(EntityFact::new(
            "occupation",
            "architect",
            0.9,
            MemoryId::new(),
        ));

        let best = profile.get_best_fact("occupation").unwrap();
        assert_eq!(best.value, "architect");
        assert_eq!(best.confidence, 0.9);
    }

    #[test]
    fn test_entity_fact_new() {
        let memory_id = MemoryId::new();
        let fact = EntityFact::new("occupation", "engineer", 0.95, memory_id.clone());

        assert_eq!(fact.fact_type, "occupation");
        assert_eq!(fact.value, "engineer");
        assert_eq!(fact.confidence, 0.95);
        assert_eq!(fact.source_memory, memory_id);
    }

    #[test]
    fn test_entity_fact_confidence_clamping() {
        let fact1 = EntityFact::new("test", "value", 1.5, MemoryId::new());
        assert_eq!(fact1.confidence, 1.0);

        let fact2 = EntityFact::new("test", "value", -0.5, MemoryId::new());
        assert_eq!(fact2.confidence, 0.0);

        let fact3 = EntityFact::new("test", "value", 0.75, MemoryId::new());
        assert_eq!(fact3.confidence, 0.75);
    }

    #[test]
    fn test_entity_profile_serialization() {
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());

        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            MemoryId::new(),
        ));
        profile.add_source_memory(MemoryId::new());

        // Serialize to JSON
        let json = serde_json::to_string(&profile).unwrap();
        assert!(json.contains("Alice"));
        assert!(json.contains("engineer"));
        assert!(json.contains("occupation"));

        // Deserialize back
        let parsed: EntityProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "Alice");
        assert_eq!(parsed.entity_type, "person");
        assert_eq!(parsed.facts.get("occupation").unwrap().len(), 1);
    }

    #[test]
    fn test_entity_fact_serialization() {
        let memory_id = MemoryId::new();
        let fact = EntityFact::new("skill", "Rust", 0.95, memory_id.clone());

        // Serialize
        let json = serde_json::to_string(&fact).unwrap();
        assert!(json.contains("Rust"));
        assert!(json.contains("skill"));

        // Deserialize
        let parsed: EntityFact = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.fact_type, "skill");
        assert_eq!(parsed.value, "Rust");
        assert_eq!(parsed.confidence, 0.95);
    }
}
