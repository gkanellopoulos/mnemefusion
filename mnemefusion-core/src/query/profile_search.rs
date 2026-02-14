//! Profile-based search for entity fact lookup (open-vocabulary)
//!
//! This module enables direct fact retrieval from Entity Profiles using
//! domain-agnostic word-overlap matching. Instead of requiring a hardcoded
//! keyword→fact_type mapping, it matches query words against ALL facts
//! (both fact_type and value text) for detected entities.
//!
//! Example: "What instrument does Caroline play?" matches against all of
//! Caroline's facts. The fact `instrument=guitar` gets a high overlap score
//! because "instrument" appears in both the query and the fact_type.
//! Similarly, "Does Caroline play guitar?" matches because "guitar" appears
//! in both the query and the fact value.

use crate::{
    error::Result,
    ingest::{EntityExtractor, SimpleEntityExtractor},
    storage::StorageEngine,
    types::MemoryId,
};
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Stop words to exclude from word-overlap matching.
/// These are too common to be meaningful signals.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
    "and", "or", "but", "not", "no", "if", "then", "that", "this", "these",
    "those", "what", "which", "who", "whom", "how", "when", "where", "why",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "them", "his", "her", "their", "s", "t", "re", "ve", "ll", "d",
];

/// Tokenize text into lowercase, stemmed words, removing stop words and punctuation.
/// Uses Porter stemming so "instruments" matches "instrument", "books" matches "book", etc.
fn tokenize(text: &str) -> HashSet<String> {
    let stemmer = Stemmer::create(Algorithm::English);
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .filter(|w| !STOP_WORDS.contains(w))
        .map(|w| stemmer.stem(w).to_string())
        .collect()
}

/// Compute word-overlap score between query words and fact text.
///
/// Returns a score in [0.0, 1.0] representing how many query words
/// appear in the fact text, normalized by query word count.
fn word_overlap_score(query_words: &HashSet<String>, fact_words: &HashSet<String>) -> f32 {
    if query_words.is_empty() {
        return 0.0;
    }
    let overlap = query_words.intersection(fact_words).count();
    if overlap == 0 {
        return 0.0;
    }
    // Normalize by the smaller set to reward high coverage
    let norm = query_words.len().min(fact_words.len()).max(1);
    (overlap as f32 / norm as f32).min(1.0)
}

/// Profile-based search engine (open-vocabulary)
///
/// Searches Entity Profiles to find memories that are sources of
/// matching facts. Uses word-overlap between query text and fact
/// text (fact_type + value) instead of a hardcoded keyword map.
pub struct ProfileSearch {
    storage: Arc<StorageEngine>,
    entity_extractor: SimpleEntityExtractor,
}

impl ProfileSearch {
    /// Create a new profile search engine
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self {
            storage,
            entity_extractor: SimpleEntityExtractor::new(),
        }
    }

    /// Search for memories based on entity profiles (open-vocabulary)
    ///
    /// This method:
    /// 1. Extracts entity names from the query
    /// 2. Looks up Entity Profiles for detected entities
    /// 3. Scores ALL facts by word-overlap between query and "{fact_type} {value}"
    /// 4. Returns source memories of matching facts with scores
    ///
    /// This is domain-agnostic: any fact_type the LLM generates (instrument,
    /// pet_name, art_style, miniature_painting, etc.) is automatically searchable
    /// without needing a hardcoded keyword mapping.
    pub fn search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        // Step 1: Extract entities from query
        let query_entities = self.entity_extractor.extract(query_text)?;

        if query_entities.is_empty() {
            return Ok(scores);
        }

        // Step 2: Tokenize the query
        let query_words = tokenize(query_text);

        if query_words.is_empty() {
            return Ok(scores);
        }

        // Step 3: Look up profiles and score all facts by word overlap
        for entity_name in &query_entities {
            if let Some(profile) = self.storage.get_entity_profile(entity_name)? {
                let mut has_any_match = false;

                // Score every fact by word overlap with query
                for (fact_type, facts) in &profile.facts {
                    for fact in facts {
                        // Build searchable text from fact_type and value
                        // Underscores in fact_type become spaces (e.g. "research_topic" → "research topic")
                        let fact_text = format!(
                            "{} {}",
                            fact_type.replace('_', " "),
                            fact.value
                        );
                        let fact_words = tokenize(&fact_text);

                        let overlap = word_overlap_score(&query_words, &fact_words);
                        if overlap > 0.0 {
                            has_any_match = true;
                            // Score = overlap * confidence
                            let fact_score = overlap * fact.confidence;
                            scores
                                .entry(fact.source_memory.clone())
                                .and_modify(|s| *s = (*s + fact_score).min(1.0))
                                .or_insert(fact_score);
                        }
                    }
                }

                // Fallback: if no fact matched by word overlap, give a small boost
                // to all source memories (entity was mentioned but no specific fact matched)
                if !has_any_match {
                    for memory_id in &profile.source_memories {
                        scores
                            .entry(memory_id.clone())
                            .and_modify(|s| *s = (*s + 0.3).min(1.0))
                            .or_insert(0.3);
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

    /// Compute open-vocabulary profile boost for a specific entity's facts
    ///
    /// Used by the entity-focused path in QueryPlanner to boost memories
    /// whose facts match the query by word overlap.
    pub fn compute_fact_boosts(
        &self,
        entity_name: &str,
        query_text: &str,
    ) -> Result<HashMap<MemoryId, f32>> {
        let mut boost_map: HashMap<MemoryId, f32> = HashMap::new();

        let query_words = tokenize(query_text);
        if query_words.is_empty() {
            return Ok(boost_map);
        }

        if let Some(profile) = self.storage.get_entity_profile(entity_name)? {
            for (fact_type, facts) in &profile.facts {
                for fact in facts {
                    let fact_text = format!(
                        "{} {}",
                        fact_type.replace('_', " "),
                        fact.value
                    );
                    let fact_words = tokenize(&fact_text);

                    let overlap = word_overlap_score(&query_words, &fact_words);
                    if overlap > 0.0 {
                        let fact_score = overlap * fact.confidence;
                        boost_map
                            .entry(fact.source_memory.clone())
                            .and_modify(|s| *s = (*s + fact_score).min(1.0))
                            .or_insert(fact_score);
                    }
                }
            }
        }

        Ok(boost_map)
    }

    /// Get entities detected in the query
    pub fn detect_entities(&self, query_text: &str) -> Result<Vec<String>> {
        self.entity_extractor.extract(query_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Unit tests for tokenize and word_overlap_score ==========

    #[test]
    fn test_tokenize_basic() {
        let words = tokenize("What instrument does Caroline play?");
        // Porter stemming: "instrument" stays, "caroline" → "carolin", "play" stays
        assert!(words.contains("instrument"));
        assert!(words.contains("carolin"));
        assert!(words.contains("play"));
        // Stop words removed
        assert!(!words.contains("what"));
        assert!(!words.contains("does"));
    }

    #[test]
    fn test_tokenize_stemming_matches_plurals() {
        // This is the key behavior: plurals/gerunds match root forms
        let query = tokenize("instruments");
        let fact = tokenize("instrument");
        assert_eq!(query, fact, "Plural should stem to same as singular");

        let query = tokenize("books");
        let fact = tokenize("book");
        assert_eq!(query, fact, "Plural should stem to same as singular");

        let query = tokenize("painted");
        let fact = tokenize("painting");
        assert_eq!(query, fact, "Past tense and gerund should stem the same");
    }

    #[test]
    fn test_tokenize_underscores_split() {
        // Underscores are now word separators (fact_type text also replaces _ with space)
        let words = tokenize("research_topic adoption agencies!");
        assert!(words.contains("research"));
        assert!(words.contains("topic"));
        // "adoption" → stemmed "adopt", "agencies" → stemmed "agenc"
        assert!(words.len() >= 3); // at least research, topic, adopt/agenc
    }

    #[test]
    fn test_word_overlap_exact_match() {
        let query = tokenize("instrument guitar");
        let fact = tokenize("instrument guitar");
        let score = word_overlap_score(&query, &fact);
        assert!((score - 1.0).abs() < 0.01, "Exact match should score 1.0, got {}", score);
    }

    #[test]
    fn test_word_overlap_partial_match() {
        let query = tokenize("instrument caroline play");
        let fact = tokenize("instrument guitar");
        let score = word_overlap_score(&query, &fact);
        // "instrument" matches, 1 overlap out of min(3, 2) = 2 → 0.5
        assert!(score > 0.0, "Should have partial overlap");
        assert!(score < 1.0, "Should not be full overlap");
    }

    #[test]
    fn test_word_overlap_no_match() {
        let query = tokenize("painting pottery");
        let fact = tokenize("instrument guitar");
        let score = word_overlap_score(&query, &fact);
        assert!((score - 0.0).abs() < 0.01, "No overlap should score 0.0");
    }

    #[test]
    fn test_word_overlap_value_match() {
        // Query mentions the VALUE, not the fact_type
        let query = tokenize("guitar caroline play");
        let fact = tokenize("instrument guitar");
        let score = word_overlap_score(&query, &fact);
        assert!(score > 0.0, "Value overlap should score > 0, got {}", score);
    }

    // ========== Integration tests with storage ==========

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
    fn test_profile_search_matches_by_fact_type_word() {
        let (storage, _dir) = create_test_storage();

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

        let search = ProfileSearch::new(storage);
        // "research" in query matches "research" in fact_type "research_topic"
        let results = search.search("What is Caroline researching?", 10).unwrap();

        assert!(!results.is_empty(), "Should find results via fact_type word overlap");
        assert!(results.contains_key(&memory_id));
    }

    #[test]
    fn test_profile_search_matches_by_value_word() {
        let (storage, _dir) = create_test_storage();

        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Caroline".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "instrument",
            "guitar",
            0.95,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        // "guitar" in query matches "guitar" in fact value
        let results = search.search("Does Caroline play guitar?", 10).unwrap();

        assert!(!results.is_empty(), "Should find results via value word overlap");
        assert!(results.contains_key(&memory_id));
    }

    #[test]
    fn test_profile_search_matches_novel_fact_types() {
        let (storage, _dir) = create_test_storage();

        // A fact type that would NEVER be in a hardcoded keyword map
        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "miniature_painting",
            "Warhammer figures",
            0.85,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        // "painting" in query matches "painting" in fact_type "miniature_painting"
        let results = search.search("Tell me about Alice painting hobby", 10).unwrap();

        assert!(!results.is_empty(), "Should find novel fact types via word overlap");
        assert!(results.contains_key(&memory_id));
    }

    #[test]
    fn test_profile_search_returns_empty_for_unknown_entity() {
        let (storage, _dir) = create_test_storage();

        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new("occupation", "engineer", 0.9, memory_id.clone()));
        profile.add_source_memory(memory_id);
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        let results = search.search("What is Bob researching?", 10).unwrap();

        assert!(results.is_empty(), "Should not find results for unknown entity");
    }

    #[test]
    fn test_profile_search_fallback_for_no_word_overlap() {
        let (storage, _dir) = create_test_storage();

        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "zodiac_sign",
            "capricorn",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        // Query has zero word overlap with "zodiac sign capricorn"
        let results = search.search("Tell me everything about Alice", 10).unwrap();

        // Should fallback to boosting all source memories
        assert!(!results.is_empty(), "Should fallback boost for entity with no fact overlap");
        assert!(results[&memory_id] < 0.5, "Fallback score should be low (0.3)");
    }

    #[test]
    fn test_profile_search_multiple_facts_ranked_by_overlap() {
        let (storage, _dir) = create_test_storage();

        let mem_guitar = MemoryId::new();
        let mem_pet = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Caroline".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new("instrument", "guitar", 0.95, mem_guitar.clone()));
        profile.add_fact(EntityFact::new("pet", "dog named Buddy", 0.9, mem_pet.clone()));
        profile.add_source_memory(mem_guitar.clone());
        profile.add_source_memory(mem_pet.clone());
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        let results = search.search("What instrument does Caroline play?", 10).unwrap();

        // "instrument" matches instrument fact, "play" doesn't match pet fact
        assert!(results.contains_key(&mem_guitar), "Should find instrument memory");
        // pet fact might or might not match depending on word overlap
        // But instrument memory should have higher score
        if results.contains_key(&mem_pet) {
            assert!(
                results[&mem_guitar] >= results[&mem_pet],
                "Instrument memory should score higher for instrument query"
            );
        }
    }

    #[test]
    fn test_compute_fact_boosts() {
        let (storage, _dir) = create_test_storage();

        let memory_id = MemoryId::new();
        let mut profile = EntityProfile::new(
            EntityId::new(),
            "Caroline".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new("instrument", "guitar", 0.95, memory_id.clone()));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        let boosts = search.compute_fact_boosts("Caroline", "What instrument does Caroline play?").unwrap();

        assert!(!boosts.is_empty(), "Should compute boosts");
        assert!(boosts.contains_key(&memory_id));
        assert!(boosts[&memory_id] > 0.0, "Boost should be positive");
    }
}
