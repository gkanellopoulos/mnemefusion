//! Output types for entity extraction
//!
//! These types represent the structured output from entity extraction,
//! guaranteed to be valid due to grammar-constrained decoding.

use crate::types::{EntityFact, MemoryId, Timestamp};
use serde::{Deserialize, Serialize};

/// Result of entity extraction - guaranteed to match schema due to grammar constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Entities mentioned in the text
    pub entities: Vec<ExtractedEntity>,

    /// Facts about entities extracted from the text
    pub entity_facts: Vec<ExtractedFact>,

    /// Main topics/themes in the text
    pub topics: Vec<String>,

    /// Importance score (0.0-1.0)
    pub importance: f32,
}

/// An entity extracted from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity name
    pub name: String,

    /// Entity type: "person", "organization", "location", "concept", "event"
    #[serde(rename = "type")]
    pub entity_type: String,
}

/// A fact about an entity extracted from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// Entity this fact is about
    pub entity: String,

    /// Fact type: occupation, research_topic, goal, preference, location, relationship, interest
    pub fact_type: String,

    /// Fact value
    pub value: String,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

impl ExtractionResult {
    /// Create an empty extraction result
    pub fn empty() -> Self {
        Self {
            entities: Vec::new(),
            entity_facts: Vec::new(),
            topics: Vec::new(),
            importance: 0.5,
        }
    }

    /// Convert entity facts to EntityFact types for storage
    ///
    /// Returns tuples of (entity_name, EntityFact) for updating entity profiles.
    pub fn to_entity_facts(&self, source_memory_id: &MemoryId) -> Vec<(String, EntityFact)> {
        self.entity_facts
            .iter()
            .map(|f| {
                (
                    f.entity.clone(),
                    EntityFact {
                        fact_type: f.fact_type.clone(),
                        value: f.value.clone(),
                        confidence: f.confidence,
                        source_memory: source_memory_id.clone(),
                        extracted_at: Timestamp::now(),
                    },
                )
            })
            .collect()
    }

    /// Get all unique entity names
    pub fn entity_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.entities.iter().map(|e| e.name.clone()).collect();
        names.sort();
        names.dedup();
        names
    }

    /// Get facts for a specific entity
    pub fn facts_for_entity(&self, entity_name: &str) -> Vec<&ExtractedFact> {
        self.entity_facts
            .iter()
            .filter(|f| f.entity.eq_ignore_ascii_case(entity_name))
            .collect()
    }
}

impl Default for ExtractionResult {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_result_parsing() {
        let json = r#"{
            "entities": [{"name": "Caroline", "type": "person"}],
            "entity_facts": [{"entity": "Caroline", "fact_type": "occupation", "value": "counselor", "confidence": 0.9}],
            "topics": ["counseling"],
            "importance": 0.8
        }"#;

        let result: ExtractionResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Caroline");
        assert_eq!(result.entity_facts[0].value, "counselor");
        assert_eq!(result.topics[0], "counseling");
        assert!((result.importance - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_entity_names() {
        let result = ExtractionResult {
            entities: vec![
                ExtractedEntity {
                    name: "Alice".to_string(),
                    entity_type: "person".to_string(),
                },
                ExtractedEntity {
                    name: "Bob".to_string(),
                    entity_type: "person".to_string(),
                },
                ExtractedEntity {
                    name: "Alice".to_string(), // duplicate
                    entity_type: "person".to_string(),
                },
            ],
            entity_facts: Vec::new(),
            topics: Vec::new(),
            importance: 0.5,
        };

        let names = result.entity_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"Alice".to_string()));
        assert!(names.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_facts_for_entity() {
        let result = ExtractionResult {
            entities: Vec::new(),
            entity_facts: vec![
                ExtractedFact {
                    entity: "Alice".to_string(),
                    fact_type: "occupation".to_string(),
                    value: "engineer".to_string(),
                    confidence: 0.9,
                },
                ExtractedFact {
                    entity: "Bob".to_string(),
                    fact_type: "location".to_string(),
                    value: "Boston".to_string(),
                    confidence: 0.8,
                },
                ExtractedFact {
                    entity: "Alice".to_string(),
                    fact_type: "interest".to_string(),
                    value: "AI".to_string(),
                    confidence: 0.7,
                },
            ],
            topics: Vec::new(),
            importance: 0.5,
        };

        let alice_facts = result.facts_for_entity("Alice");
        assert_eq!(alice_facts.len(), 2);
        assert!(alice_facts.iter().any(|f| f.fact_type == "occupation"));
        assert!(alice_facts.iter().any(|f| f.fact_type == "interest"));
    }

    #[test]
    fn test_empty_result() {
        let result = ExtractionResult::empty();
        assert!(result.entities.is_empty());
        assert!(result.entity_facts.is_empty());
        assert!(result.topics.is_empty());
        assert!((result.importance - 0.5).abs() < 0.01);
    }
}
