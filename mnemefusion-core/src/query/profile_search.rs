//! Profile-based search for entity fact lookup
//!
//! This module enables direct fact retrieval from Entity Profiles.
//! When a query asks "What is Caroline researching?", we can look up
//! Caroline's profile and find the research_topic fact directly.

use crate::{
    error::Result,
    ingest::{EntityExtractor, SimpleEntityExtractor},
    storage::StorageEngine,
    types::MemoryId,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Detects fact types from query keywords
///
/// Maps natural language query patterns to fact_type values
/// stored in Entity Profiles.
pub struct FactTypeDetector {
    /// Mapping of keywords to fact types
    keyword_mappings: Vec<(Vec<&'static str>, &'static str)>,
}

impl Default for FactTypeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FactTypeDetector {
    /// Create a new fact type detector with default mappings
    pub fn new() -> Self {
        Self {
            keyword_mappings: vec![
                // Research/Study patterns
                (
                    vec![
                        "research",
                        "researching",
                        "studying",
                        "investigating",
                        "looking into",
                        "exploring",
                        "working on",
                    ],
                    "research_topic",
                ),
                // Occupation patterns
                (
                    vec![
                        "work",
                        "works",
                        "working",
                        "job",
                        "occupation",
                        "profession",
                        "career",
                        "employed",
                        "does for a living",
                    ],
                    "occupation",
                ),
                // Location patterns
                (
                    vec![
                        "live",
                        "lives",
                        "living",
                        "located",
                        "location",
                        "where",
                        "based in",
                        "from",
                        "residence",
                    ],
                    "location",
                ),
                // Preference patterns
                (
                    vec![
                        "like",
                        "likes",
                        "prefer",
                        "prefers",
                        "favorite",
                        "favourite",
                        "enjoy",
                        "enjoys",
                        "love",
                        "loves",
                    ],
                    "preference",
                ),
                // Goal patterns
                (
                    vec![
                        "goal",
                        "goals",
                        "want",
                        "wants",
                        "trying to",
                        "plan",
                        "planning",
                        "intend",
                        "aim",
                        "objective",
                    ],
                    "goal",
                ),
                // Relationship patterns
                (
                    vec![
                        "relationship",
                        "know",
                        "knows",
                        "friend",
                        "married",
                        "partner",
                        "sibling",
                        "parent",
                        "child",
                        "colleague",
                    ],
                    "relationship",
                ),
                // Skill patterns
                (
                    vec![
                        "skill",
                        "skills",
                        "good at",
                        "expert",
                        "proficient",
                        "capable",
                        "ability",
                        "talent",
                    ],
                    "skill",
                ),
                // Education patterns
                (
                    vec![
                        "study",
                        "studied",
                        "degree",
                        "graduate",
                        "university",
                        "school",
                        "education",
                        "major",
                    ],
                    "education",
                ),
            ],
        }
    }

    /// Detect fact types from query text
    ///
    /// Returns a list of fact types that match keywords in the query.
    /// Multiple fact types may be returned if the query is ambiguous.
    ///
    /// # Arguments
    ///
    /// * `query_text` - The query text to analyze
    ///
    /// # Returns
    ///
    /// Vector of fact type strings that may be relevant
    pub fn detect(&self, query_text: &str) -> Vec<&'static str> {
        let query_lower = query_text.to_lowercase();
        let mut detected = Vec::new();

        for (keywords, fact_type) in &self.keyword_mappings {
            for keyword in keywords {
                if query_lower.contains(keyword) {
                    if !detected.contains(fact_type) {
                        detected.push(*fact_type);
                    }
                    break; // Found match for this fact_type, move to next
                }
            }
        }

        detected
    }

    /// Get all known fact types
    pub fn known_fact_types(&self) -> Vec<&'static str> {
        self.keyword_mappings
            .iter()
            .map(|(_, fact_type)| *fact_type)
            .collect()
    }
}

/// Profile-based search engine
///
/// Searches Entity Profiles to find memories that are sources of
/// matching facts for detected entities and fact types.
pub struct ProfileSearch {
    storage: Arc<StorageEngine>,
    entity_extractor: SimpleEntityExtractor,
    fact_detector: FactTypeDetector,
}

impl ProfileSearch {
    /// Create a new profile search engine
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self {
            storage,
            entity_extractor: SimpleEntityExtractor::new(),
            fact_detector: FactTypeDetector::new(),
        }
    }

    /// Search for memories based on entity profiles
    ///
    /// This method:
    /// 1. Extracts entity names from the query
    /// 2. Detects relevant fact types from query keywords
    /// 3. Looks up Entity Profiles for detected entities
    /// 4. Returns source memories of matching facts with scores
    ///
    /// # Arguments
    ///
    /// * `query_text` - The query text
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    ///
    /// HashMap of MemoryId to score for memories that are sources of matching facts
    pub fn search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        // Step 1: Extract entities from query
        let query_entities = self.entity_extractor.extract(query_text)?;

        if query_entities.is_empty() {
            return Ok(scores);
        }

        // Step 2: Detect relevant fact types from query
        let fact_types = self.fact_detector.detect(query_text);

        // Step 3: Look up profiles for each entity
        for entity_name in &query_entities {
            if let Some(profile) = self.storage.get_entity_profile(entity_name)? {
                // Step 4: Find matching facts
                if fact_types.is_empty() {
                    // No specific fact type detected - boost all source memories
                    for memory_id in &profile.source_memories {
                        scores
                            .entry(memory_id.clone())
                            .and_modify(|s| *s = (*s + 0.5).min(1.0))
                            .or_insert(0.5);
                    }
                } else {
                    // Specific fact types detected - boost memories with those facts
                    for fact_type in &fact_types {
                        let facts = profile.get_facts(fact_type);
                        for fact in facts {
                            // Score based on fact confidence
                            let fact_score = fact.confidence;
                            scores
                                .entry(fact.source_memory.clone())
                                .and_modify(|s| *s = (*s + fact_score).min(1.0))
                                .or_insert(fact_score);
                        }
                    }
                }
            }
        }

        // Limit results
        if scores.len() > limit {
            let mut sorted: Vec<_> = scores.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores = sorted.into_iter().take(limit).collect();
        }

        Ok(scores)
    }

    /// Get entities detected in the query
    ///
    /// Useful for debugging and understanding what entities were found.
    pub fn detect_entities(&self, query_text: &str) -> Result<Vec<String>> {
        self.entity_extractor.extract(query_text)
    }

    /// Get fact types detected in the query
    ///
    /// Useful for debugging and understanding what fact types were matched.
    pub fn detect_fact_types(&self, query_text: &str) -> Vec<&'static str> {
        self.fact_detector.detect(query_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_type_detector_research() {
        let detector = FactTypeDetector::new();

        assert!(detector.detect("What is Alice researching?").contains(&"research_topic"));
        assert!(detector.detect("What is Bob studying?").contains(&"research_topic"));
        assert!(detector.detect("investigating something").contains(&"research_topic"));
    }

    #[test]
    fn test_fact_type_detector_occupation() {
        let detector = FactTypeDetector::new();

        assert!(detector.detect("Where does Alice work?").contains(&"occupation"));
        assert!(detector.detect("What is Bob's job?").contains(&"occupation"));
        assert!(detector.detect("What is their occupation?").contains(&"occupation"));
    }

    #[test]
    fn test_fact_type_detector_location() {
        let detector = FactTypeDetector::new();

        assert!(detector.detect("Where does Alice live?").contains(&"location"));
        assert!(detector.detect("Where is Bob located?").contains(&"location"));
    }

    #[test]
    fn test_fact_type_detector_preference() {
        let detector = FactTypeDetector::new();

        assert!(detector.detect("What does Alice like?").contains(&"preference"));
        assert!(detector.detect("What is Bob's favorite color?").contains(&"preference"));
    }

    #[test]
    fn test_fact_type_detector_goal() {
        let detector = FactTypeDetector::new();

        assert!(detector.detect("What is Alice's goal?").contains(&"goal"));
        assert!(detector.detect("What does Bob want to do?").contains(&"goal"));
        assert!(detector.detect("What is she trying to achieve?").contains(&"goal"));
    }

    #[test]
    fn test_fact_type_detector_multiple() {
        let detector = FactTypeDetector::new();

        // Query with multiple fact types
        let fact_types = detector.detect("Where does Alice work and live?");
        assert!(fact_types.contains(&"occupation"));
        assert!(fact_types.contains(&"location"));
    }

    #[test]
    fn test_fact_type_detector_no_match() {
        let detector = FactTypeDetector::new();

        let fact_types = detector.detect("Hello world");
        assert!(fact_types.is_empty());
    }

    #[test]
    fn test_fact_type_detector_case_insensitive() {
        let detector = FactTypeDetector::new();

        assert!(detector.detect("WHAT IS ALICE RESEARCHING?").contains(&"research_topic"));
        assert!(detector.detect("Where Does Bob LIVE?").contains(&"location"));
    }

    // ========== Integration Tests with Storage ==========

    use crate::storage::StorageEngine;
    use crate::types::{EntityFact, EntityId, EntityProfile, MemoryId};
    use tempfile::tempdir;

    fn create_test_storage() -> (Arc<StorageEngine>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let storage = Arc::new(StorageEngine::open(&path).unwrap());
        (storage, dir)
    }

    #[test]
    fn test_profile_search_finds_research_topic() {
        let (storage, _dir) = create_test_storage();

        // Create a profile for "Caroline" with a research_topic fact
        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Caroline".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "research_topic",
            "adoption agencies",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        // Create ProfileSearch and query
        let search = ProfileSearch::new(storage);
        let results = search.search("What is Caroline researching?", 10).unwrap();

        // Should find the memory with the research_topic fact
        assert!(!results.is_empty(), "Should find profile results");
        assert!(results.contains_key(&memory_id), "Should find the source memory");
        assert!(results[&memory_id] > 0.8, "Should have high score for matching fact");
    }

    #[test]
    fn test_profile_search_returns_empty_for_no_matching_entity() {
        let (storage, _dir) = create_test_storage();

        // Create a profile for "Alice"
        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id);
        storage.store_entity_profile(&profile).unwrap();

        // Query for a different entity
        let search = ProfileSearch::new(storage);
        let results = search.search("What is Bob researching?", 10).unwrap();

        // Should return empty - no profile for Bob
        assert!(results.is_empty(), "Should not find results for unknown entity");
    }

    #[test]
    fn test_profile_search_returns_empty_for_no_matching_fact_type() {
        let (storage, _dir) = create_test_storage();

        // Create a profile with occupation fact
        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        // Query for research_topic (different fact type)
        let search = ProfileSearch::new(storage);
        let results = search.search("What is Alice researching?", 10).unwrap();

        // Should return empty - no research_topic facts
        assert!(results.is_empty(), "Should not find results for unmatched fact type");
    }

    #[test]
    fn test_profile_search_multiple_facts_same_entity() {
        let (storage, _dir) = create_test_storage();

        // Create a profile with multiple research_topic facts
        let memory_id1 = MemoryId::new();
        let memory_id2 = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Caroline".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "research_topic",
            "adoption agencies",
            0.9,
            memory_id1.clone(),
        ));
        profile.add_fact(EntityFact::new(
            "research_topic",
            "biological parents",
            0.85,
            memory_id2.clone(),
        ));
        profile.add_source_memory(memory_id1.clone());
        profile.add_source_memory(memory_id2.clone());
        storage.store_entity_profile(&profile).unwrap();

        // Query for research
        let search = ProfileSearch::new(storage);
        let results = search.search("What is Caroline researching?", 10).unwrap();

        // Should find both source memories
        assert_eq!(results.len(), 2, "Should find both source memories");
        assert!(results.contains_key(&memory_id1));
        assert!(results.contains_key(&memory_id2));

        // Higher confidence fact should have higher score
        assert!(results[&memory_id1] >= results[&memory_id2],
            "Higher confidence fact should have higher score");
    }

    #[test]
    fn test_profile_search_case_insensitive_entity_lookup() {
        let (storage, _dir) = create_test_storage();

        // Create profile with mixed case name
        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Caroline".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "research_topic",
            "adoption",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        // Query with different case
        let search = ProfileSearch::new(storage);
        let results = search.search("What is CAROLINE researching?", 10).unwrap();

        // Should find the profile (case-insensitive lookup)
        assert!(!results.is_empty(), "Should find profile with case-insensitive lookup");
        assert!(results.contains_key(&memory_id));
    }

    #[test]
    fn test_profile_search_generic_query_boosts_all_sources() {
        let (storage, _dir) = create_test_storage();

        // Create profile with source memories but no detectable fact types in query
        let memory_id1 = MemoryId::new();
        let memory_id2 = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "custom_fact",
            "value1",
            0.9,
            memory_id1.clone(),
        ));
        profile.add_source_memory(memory_id1.clone());
        profile.add_source_memory(memory_id2.clone());
        storage.store_entity_profile(&profile).unwrap();

        // Query without matching fact type keywords
        let search = ProfileSearch::new(storage);
        let results = search.search("Tell me about Alice", 10).unwrap();

        // Should boost all source memories
        assert_eq!(results.len(), 2, "Should find all source memories for generic query");
    }
}
