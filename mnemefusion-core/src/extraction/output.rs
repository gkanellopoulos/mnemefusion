//! Output types for entity extraction
//!
//! These types represent the structured output from entity extraction,
//! guaranteed to be valid due to grammar-constrained decoding.

use crate::types::{EntityFact, MemoryId, Timestamp};
use serde::{Deserialize, Serialize};

/// Result of entity extraction
///
/// All fields have serde defaults to handle partial model outputs gracefully.
/// Different models may omit fields — defaults ensure parsing never fails
/// due to missing optional data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Entities mentioned in the text
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,

    /// Facts about entities extracted from the text
    #[serde(default)]
    pub entity_facts: Vec<ExtractedFact>,

    /// Main topics/themes in the text
    #[serde(default)]
    pub topics: Vec<String>,

    /// Importance score (0.0-1.0)
    #[serde(default = "default_importance")]
    pub importance: f32,

    /// Typed sub-records decomposed from the original text (ENGRAM-inspired).
    /// Each record is a self-contained sentence tagged as episodic, semantic, or procedural.
    /// Empty for legacy extractions (backward compatible via serde default).
    #[serde(default)]
    pub records: Vec<TypedRecord>,

    /// Entity-to-entity relationships extracted from the text.
    /// E.g., "spouse", "sibling", "colleague" links between named entities.
    /// Empty for legacy extractions (backward compatible via serde default).
    #[serde(default)]
    pub relationships: Vec<ExtractedRelationship>,
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

/// A typed sub-record decomposed from a conversation turn (ENGRAM-inspired).
///
/// Each record is a self-contained sentence that can be embedded independently.
/// Records are tagged by type: episodic (events), semantic (stable facts),
/// or procedural (routines/instructions).
///
/// Research basis: ENGRAM (arXiv 2511.12960) demonstrates significant gains from
/// typed memory separation. TReMu (ACL 2025) uses inferred event dates for
/// temporal QA improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedRecord {
    /// Record type: "episodic", "semantic", or "procedural"
    pub record_type: String,

    /// Self-contained summary sentence (embeddable independently)
    pub summary: String,

    /// ISO-8601 date for episodic records (inferred from relative expressions + session date).
    /// None for semantic/procedural records.
    #[serde(default)]
    pub event_date: Option<String>,

    /// Entity names involved in this record
    #[serde(default)]
    pub entities: Vec<String>,
}

/// An extracted relationship between two entities.
///
/// Represents directional entity-to-entity links like "spouse", "sibling",
/// "colleague", etc. Used for multi-hop graph traversal at query time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    /// Source entity name
    pub from_entity: String,

    /// Target entity name
    pub to_entity: String,

    /// Relationship type (e.g., "spouse", "sibling", "colleague", "friend")
    pub relation_type: String,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,
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

    /// Confidence score (0.0-1.0). Defaults to 0.9 when omitted by the model.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    0.9
}

fn default_importance() -> f32 {
    0.5
}

impl ExtractionResult {
    /// Create an empty extraction result
    pub fn empty() -> Self {
        Self {
            entities: Vec::new(),
            entity_facts: Vec::new(),
            topics: Vec::new(),
            importance: 0.5,
            records: Vec::new(),
            relationships: Vec::new(),
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
            ..Default::default()
        };

        let names = result.entity_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"Alice".to_string()));
        assert!(names.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_facts_for_entity() {
        let result = ExtractionResult {
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
            ..Default::default()
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
        assert!(result.records.is_empty());
        assert!(result.relationships.is_empty());
        assert!((result.importance - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_typed_record_serde() {
        let record = TypedRecord {
            record_type: "episodic".to_string(),
            summary: "Alice went hiking last weekend".to_string(),
            event_date: Some("2023-03-11".to_string()),
            entities: vec!["Alice".to_string()],
        };

        let json = serde_json::to_string(&record).unwrap();
        let parsed: TypedRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.record_type, "episodic");
        assert_eq!(parsed.summary, "Alice went hiking last weekend");
        assert_eq!(parsed.event_date, Some("2023-03-11".to_string()));
        assert_eq!(parsed.entities, vec!["Alice"]);
    }

    #[test]
    fn test_typed_record_optional_fields() {
        // event_date and entities should default to None/empty when absent
        let json = r#"{"record_type": "semantic", "summary": "Alice works at Google"}"#;
        let record: TypedRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.record_type, "semantic");
        assert!(record.event_date.is_none());
        assert!(record.entities.is_empty());
    }

    #[test]
    fn test_extracted_relationship_serde() {
        let rel = ExtractedRelationship {
            from_entity: "Alice".to_string(),
            to_entity: "Bob".to_string(),
            relation_type: "spouse".to_string(),
            confidence: 0.95,
        };

        let json = serde_json::to_string(&rel).unwrap();
        let parsed: ExtractedRelationship = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.from_entity, "Alice");
        assert_eq!(parsed.to_entity, "Bob");
        assert_eq!(parsed.relation_type, "spouse");
        assert!((parsed.confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_backward_compat_legacy_json() {
        // Legacy JSON without records/relationships fields should parse fine
        let json = r#"{
            "entities": [{"name": "Alice", "type": "person"}],
            "entity_facts": [],
            "topics": ["work"],
            "importance": 0.7
        }"#;

        let result: ExtractionResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.entities.len(), 1);
        assert!(result.records.is_empty());
        assert!(result.relationships.is_empty());
    }

    #[test]
    fn test_full_extraction_with_records_and_relationships() {
        let json = r#"{
            "entities": [{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}],
            "entity_facts": [{"entity": "Alice", "fact_type": "occupation", "value": "engineer", "confidence": 0.9}],
            "topics": ["work"],
            "importance": 0.8,
            "records": [
                {"record_type": "semantic", "summary": "Alice works as an engineer", "entities": ["Alice"]},
                {"record_type": "episodic", "summary": "Alice met Bob at a conference last week", "event_date": "2023-03-08", "entities": ["Alice", "Bob"]}
            ],
            "relationships": [
                {"from_entity": "Alice", "to_entity": "Bob", "relation_type": "colleague", "confidence": 0.85}
            ]
        }"#;

        let result: ExtractionResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.records.len(), 2);
        assert_eq!(result.records[0].record_type, "semantic");
        assert!(result.records[0].event_date.is_none());
        assert_eq!(result.records[1].record_type, "episodic");
        assert_eq!(result.records[1].event_date, Some("2023-03-08".to_string()));
        assert_eq!(result.relationships.len(), 1);
        assert_eq!(result.relationships[0].relation_type, "colleague");
    }
}
