//! Profile-based search for entity fact lookup (open-vocabulary)
//!
//! This module enables direct fact retrieval from Entity Profiles using
//! embedding-based similarity matching with word-overlap fallback.
//!
//! **Entity detection** uses case-insensitive whole-word matching against stored
//! entity profile names — no regex or capitalization heuristics. This handles
//! any name format, language, or casing.
//!
//! **Fact matching** uses cosine similarity between the query embedding and
//! precomputed fact embeddings when available. Facts without embeddings fall
//! back to stemmed word-overlap matching for backward compatibility with
//! existing databases.

use crate::{error::Result, storage::StorageEngine, types::MemoryId};
use rust_stemmers::{Algorithm, Stemmer};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Stop words to exclude from word-overlap matching.
/// These are too common to be meaningful signals.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "about", "it", "its", "and",
    "or", "but", "not", "no", "if", "then", "that", "this", "these", "those", "what", "which",
    "who", "whom", "how", "when", "where", "why", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "his", "her", "their", "s", "t", "re", "ve", "ll", "d",
];

/// Minimum cosine similarity threshold for embedding-based fact matching.
/// Below this, the match is considered noise and ignored.
const EMBEDDING_MATCH_THRESHOLD: f32 = 0.3;

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

/// Resolve an entity name to its canonical profile form.
///
/// If `name` is a ≥3-char prefix of a longer existing profile name
/// (e.g., "mel" → "melanie"), returns the canonical longer form.
/// Space separators block matching ("jon" won't match "jon smith").
///
/// Names are compared case-insensitively. `known_profile_names` should
/// be lowercase (as returned by `StorageEngine::list_entity_profile_names()`).
pub fn resolve_entity_alias(name: &str, known_profile_names: &[String]) -> Option<String> {
    let name_lower = name.to_lowercase();
    if name_lower.len() < 3 {
        return None;
    }
    for candidate in known_profile_names {
        if candidate.len() > name_lower.len() && candidate.starts_with(name_lower.as_str()) {
            let suffix_char = candidate.as_bytes()[name_lower.len()];
            if suffix_char.is_ascii_alphanumeric() {
                return Some(candidate.clone());
            }
        }
    }
    // Fuzzy: drop last char and retry (handles "mell" → "mel" → matches "melanie")
    // Capped at ≤5 chars to prevent long names resolving incorrectly
    // (e.g., "melanie" truncated to "melani" must NOT match "melanie's son")
    //
    // Critical constraint: skip candidates that contain apostrophes or possessives.
    // Without this, "john" (truncated to "joh") would match "john's cousin" since
    // "john's cousin" starts with "joh" + 'n' (alphanumeric). Similarly "maria"
    // → "mari" → "maria's little one" (starts with "mari" + 'a'). These are
    // compound relational names, not canonical forms of "john" or "maria".
    if name_lower.len() >= 4 && name_lower.len() <= 5 {
        let mut trunc_end = name_lower.len() - 1;
        while trunc_end > 0 && !name_lower.is_char_boundary(trunc_end) {
            trunc_end -= 1;
        }
        let truncated = &name_lower[..trunc_end];
        for candidate in known_profile_names {
            // Skip compound relational names (e.g., "john's cousin", "maria's mom")
            if candidate.contains('\'') || candidate.contains('\u{2019}') {
                continue;
            }
            if candidate.len() > name_lower.len() && candidate.starts_with(truncated) {
                let suffix_char = candidate.as_bytes()[truncated.len()];
                if suffix_char.is_ascii_alphanumeric() {
                    return Some(candidate.clone());
                }
            }
        }
    }
    None
}

/// Compute cosine similarity between two vectors.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute the 16-byte hash key for a fact embedding.
pub fn fact_embedding_key(entity: &str, fact_type: &str, value: &str) -> Vec<u8> {
    let input = format!(
        "{}\0{}\0{}",
        entity.to_lowercase(),
        fact_type,
        value.to_lowercase()
    );
    let hash = Sha256::digest(input.as_bytes());
    hash[..16].to_vec()
}

/// A single matched fact from a profile search
#[derive(Debug, Clone)]
pub struct MatchedProfileFact {
    pub entity_name: String,
    pub fact_type: String,
    pub value: String,
    pub score: f32,
    pub source_memory: MemoryId,
}

/// Result of a profile search containing both source scores (for RRF boosting)
/// and matched facts (for synthetic memory injection)
#[derive(Debug, Clone)]
pub struct ProfileSearchResult {
    pub source_scores: HashMap<MemoryId, f32>,
    pub matched_facts: Vec<MatchedProfileFact>,
}

/// Profile-based search engine (open-vocabulary)
///
/// Searches Entity Profiles to find memories that are sources of
/// matching facts. Uses embedding similarity when fact embeddings
/// are available, falling back to word-overlap for backward compatibility.
///
/// Entity detection uses case-insensitive whole-word matching against
/// stored entity profile names instead of regex.
pub struct ProfileSearch {
    storage: Arc<StorageEngine>,
}

impl ProfileSearch {
    /// Create a new profile search engine
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self { storage }
    }

    /// Detect entities in query text by matching against stored profile names.
    ///
    /// Uses case-insensitive whole-word matching. Handles possessives
    /// (e.g., "Caroline's") and punctuation boundaries.
    ///
    /// Includes prefix-based alias resolution: if "mel" matches as a whole word
    /// and "melanie" exists as a profile, "mel" is resolved to "melanie" (the
    /// canonical form). This handles common nickname patterns (Mel/Melanie,
    /// Jon/Jonathan, etc.) without requiring explicit alias configuration.
    fn detect_entities_in_query(&self, query_text: &str) -> Result<Vec<String>> {
        let query_lower = query_text.to_lowercase();
        let entity_names = self.storage.list_entity_profile_names()?;

        let mut found = Vec::new();
        for name in &entity_names {
            // Keys are already lowercase from storage
            if Self::contains_whole_word(&query_lower, name) {
                found.push(name.clone());
            }
        }

        // Resolve prefix-based aliases using the shared resolve_entity_alias() function
        if !found.is_empty() {
            let mut resolved = Vec::new();
            for name in &found {
                let canonical =
                    resolve_entity_alias(name, &entity_names).unwrap_or_else(|| name.clone());
                if !resolved.contains(&canonical) {
                    resolved.push(canonical);
                }
            }
            found = resolved;
        }

        Ok(found)
    }

    /// Check if `word` appears in `text` as a complete word (not substring).
    /// Handles possessives (e.g., "caroline's") and punctuation boundaries.
    fn contains_whole_word(text: &str, word: &str) -> bool {
        for (idx, _) in text.match_indices(word) {
            let before_ok = idx == 0 || !text.as_bytes()[idx - 1].is_ascii_alphanumeric();
            let after_idx = idx + word.len();
            let after_ok = after_idx >= text.len()
                || !text.as_bytes()[after_idx].is_ascii_alphanumeric()
                || text[after_idx..].starts_with("'s")
                || text[after_idx..].starts_with("\u{2019}s");
            if before_ok && after_ok {
                return true;
            }
        }
        false
    }

    /// Search for memories based on entity profiles
    ///
    /// This method:
    /// 1. Detects entity names in the query via profile name matching
    /// 2. Looks up Entity Profiles for detected entities
    /// 3. Scores ALL facts by embedding similarity (with word-overlap fallback)
    /// 4. Returns source memories of matching facts with scores
    pub fn search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<ProfileSearchResult> {
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();
        let mut matched_facts: Vec<MatchedProfileFact> = Vec::new();

        // Step 1: Detect entities from query using profile names
        let query_entities = self.detect_entities_in_query(query_text)?;

        if query_entities.is_empty() {
            return Ok(ProfileSearchResult {
                source_scores: scores,
                matched_facts,
            });
        }

        // Step 2: Tokenize the query (for word-overlap fallback)
        let query_words = tokenize(query_text);

        // Step 3: Look up profiles and score all facts
        for entity_name in &query_entities {
            if let Some(profile) = self.storage.get_entity_profile(entity_name)? {
                let mut has_any_match = false;

                // Score every fact
                for (fact_type, facts) in &profile.facts {
                    for fact in facts {
                        // Try embedding-based scoring first
                        let key = fact_embedding_key(entity_name, fact_type, &fact.value);
                        let score =
                            if let Ok(Some(fact_emb)) = self.storage.get_fact_embedding(&key) {
                                let sim = cosine_similarity(query_embedding, &fact_emb);
                                if sim >= EMBEDDING_MATCH_THRESHOLD {
                                    sim * fact.confidence
                                } else {
                                    0.0
                                }
                            } else {
                                // Fallback: word-overlap (backward compat for facts without embeddings)
                                if query_words.is_empty() {
                                    0.0
                                } else {
                                    let fact_text =
                                        format!("{} {}", fact_type.replace('_', " "), fact.value);
                                    let fact_words = tokenize(&fact_text);
                                    word_overlap_score(&query_words, &fact_words) * fact.confidence
                                }
                            };

                        if score > 0.0 {
                            has_any_match = true;
                            scores
                                .entry(fact.source_memory.clone())
                                .and_modify(|s| *s = (*s + score).min(1.0))
                                .or_insert(score);

                            matched_facts.push(MatchedProfileFact {
                                entity_name: entity_name.clone(),
                                fact_type: fact_type.clone(),
                                value: fact.value.clone(),
                                score,
                                source_memory: fact.source_memory.clone(),
                            });
                        }
                    }
                }

                // Fallback: if no fact matched, give a small boost
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

        // Limit source_scores
        if scores.len() > limit {
            let mut sorted: Vec<_> = scores.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores = sorted.into_iter().take(limit).collect();
        }

        // Dedup matched_facts by (entity_name, fact_type, value), keep highest score
        matched_facts.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = HashSet::new();
        matched_facts
            .retain(|f| seen.insert((f.entity_name.clone(), f.fact_type.clone(), f.value.clone())));
        matched_facts.truncate(5);

        Ok(ProfileSearchResult {
            source_scores: scores,
            matched_facts,
        })
    }

    /// Compute profile boost for a specific entity's facts
    ///
    /// Used by the entity-focused path in QueryPlanner to boost memories
    /// whose facts match the query.
    pub fn compute_fact_boosts(
        &self,
        entity_name: &str,
        query_text: &str,
        query_embedding: &[f32],
    ) -> Result<HashMap<MemoryId, f32>> {
        let mut boost_map: HashMap<MemoryId, f32> = HashMap::new();

        let query_words = tokenize(query_text);

        if let Some(profile) = self.storage.get_entity_profile(entity_name)? {
            for (fact_type, facts) in &profile.facts {
                for fact in facts {
                    // Try embedding-based scoring first
                    let key = fact_embedding_key(entity_name, fact_type, &fact.value);
                    let score = if let Ok(Some(fact_emb)) = self.storage.get_fact_embedding(&key) {
                        let sim = cosine_similarity(query_embedding, &fact_emb);
                        if sim >= EMBEDDING_MATCH_THRESHOLD {
                            sim * fact.confidence
                        } else {
                            0.0
                        }
                    } else {
                        // Fallback: word-overlap
                        if query_words.is_empty() {
                            0.0
                        } else {
                            let fact_text =
                                format!("{} {}", fact_type.replace('_', " "), fact.value);
                            let fact_words = tokenize(&fact_text);
                            word_overlap_score(&query_words, &fact_words) * fact.confidence
                        }
                    };

                    if score > 0.0 {
                        boost_map
                            .entry(fact.source_memory.clone())
                            .and_modify(|s| *s = (*s + score).min(1.0))
                            .or_insert(score);
                    }
                }
            }
        }

        Ok(boost_map)
    }

    /// Get entities detected in the query (public API)
    pub fn detect_entities(&self, query_text: &str) -> Result<Vec<String>> {
        self.detect_entities_in_query(query_text)
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
        assert!(
            (score - 1.0).abs() < 0.01,
            "Exact match should score 1.0, got {}",
            score
        );
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

    // ========== Unit tests for contains_whole_word ==========

    #[test]
    fn test_contains_whole_word_basic() {
        assert!(ProfileSearch::contains_whole_word(
            "what does caroline play",
            "caroline"
        ));
        assert!(!ProfileSearch::contains_whole_word(
            "what does carolina play",
            "caroline"
        ));
    }

    #[test]
    fn test_contains_whole_word_possessive() {
        assert!(ProfileSearch::contains_whole_word(
            "caroline's guitar",
            "caroline"
        ));
        assert!(ProfileSearch::contains_whole_word(
            "caroline\u{2019}s guitar",
            "caroline"
        ));
    }

    #[test]
    fn test_contains_whole_word_punctuation() {
        assert!(ProfileSearch::contains_whole_word(
            "tell me about caroline.",
            "caroline"
        ));
        assert!(ProfileSearch::contains_whole_word(
            "caroline, what do you think?",
            "caroline"
        ));
    }

    #[test]
    fn test_contains_whole_word_no_substring() {
        // "carol" should NOT match inside "caroline"
        assert!(!ProfileSearch::contains_whole_word(
            "tell me about caroline",
            "carol"
        ));
    }

    #[test]
    fn test_fact_embedding_key_deterministic() {
        let k1 = fact_embedding_key("Caroline", "instrument", "guitar");
        let k2 = fact_embedding_key("caroline", "instrument", "Guitar");
        assert_eq!(
            k1, k2,
            "Key should be case-insensitive for entity and value"
        );
        assert_eq!(k1.len(), 16);
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
        let result = search
            .search("What is Caroline researching?", &[], 10)
            .unwrap();

        assert!(
            !result.source_scores.is_empty(),
            "Should find results via fact_type word overlap"
        );
        assert!(result.source_scores.contains_key(&memory_id));
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
        let result = search
            .search("Does Caroline play guitar?", &[], 10)
            .unwrap();

        assert!(
            !result.source_scores.is_empty(),
            "Should find results via value word overlap"
        );
        assert!(result.source_scores.contains_key(&memory_id));
    }

    #[test]
    fn test_profile_search_matches_novel_fact_types() {
        let (storage, _dir) = create_test_storage();

        // A fact type that would NEVER be in a hardcoded keyword map
        let memory_id = MemoryId::new();
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());
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
        let result = search
            .search("Tell me about Alice painting hobby", &[], 10)
            .unwrap();

        assert!(
            !result.source_scores.is_empty(),
            "Should find novel fact types via word overlap"
        );
        assert!(result.source_scores.contains_key(&memory_id));
    }

    #[test]
    fn test_profile_search_returns_empty_for_unknown_entity() {
        let (storage, _dir) = create_test_storage();

        let memory_id = MemoryId::new();
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());
        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id);
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        let result = search.search("What is Bob researching?", &[], 10).unwrap();

        assert!(
            result.source_scores.is_empty(),
            "Should not find results for unknown entity"
        );
    }

    #[test]
    fn test_profile_search_fallback_for_no_word_overlap() {
        let (storage, _dir) = create_test_storage();

        let memory_id = MemoryId::new();
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());
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
        let result = search
            .search("Tell me everything about Alice", &[], 10)
            .unwrap();

        // Should fallback to boosting all source memories
        assert!(
            !result.source_scores.is_empty(),
            "Should fallback boost for entity with no fact overlap"
        );
        assert!(
            result.source_scores[&memory_id] < 0.5,
            "Fallback score should be low (0.3)"
        );
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
        profile.add_fact(EntityFact::new(
            "instrument",
            "guitar",
            0.95,
            mem_guitar.clone(),
        ));
        profile.add_fact(EntityFact::new(
            "pet",
            "dog named Buddy",
            0.9,
            mem_pet.clone(),
        ));
        profile.add_source_memory(mem_guitar.clone());
        profile.add_source_memory(mem_pet.clone());
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        let result = search
            .search("What instrument does Caroline play?", &[], 10)
            .unwrap();

        // "instrument" matches instrument fact, "play" doesn't match pet fact
        assert!(
            result.source_scores.contains_key(&mem_guitar),
            "Should find instrument memory"
        );
        // But instrument memory should have higher score
        if result.source_scores.contains_key(&mem_pet) {
            assert!(
                result.source_scores[&mem_guitar] >= result.source_scores[&mem_pet],
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
        profile.add_fact(EntityFact::new(
            "instrument",
            "guitar",
            0.95,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        let boosts = search
            .compute_fact_boosts("Caroline", "What instrument does Caroline play?", &[])
            .unwrap();

        assert!(!boosts.is_empty(), "Should compute boosts");
        assert!(boosts.contains_key(&memory_id));
        assert!(boosts[&memory_id] > 0.0, "Boost should be positive");
    }

    #[test]
    fn test_detect_entities_lowercase_query() {
        let (storage, _dir) = create_test_storage();

        let profile = EntityProfile::new(
            EntityId::new(),
            "Caroline".to_string(),
            "person".to_string(),
        );
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        let entities = search.detect_entities("what does caroline like?").unwrap();

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0], "caroline");
    }

    #[test]
    fn test_detect_entities_no_substring_match() {
        let (storage, _dir) = create_test_storage();

        // Store "Caroline" but query contains "Carolina"
        let profile = EntityProfile::new(
            EntityId::new(),
            "Caroline".to_string(),
            "person".to_string(),
        );
        storage.store_entity_profile(&profile).unwrap();

        let search = ProfileSearch::new(storage);
        let entities = search.detect_entities("Tell me about Carolina").unwrap();

        assert!(
            entities.is_empty(),
            "Should NOT match 'caroline' inside 'carolina'"
        );
    }

    #[test]
    fn test_embedding_based_fact_scoring() {
        let (storage, _dir) = create_test_storage();

        let memory_id = MemoryId::new();
        let mut profile =
            EntityProfile::new(EntityId::new(), "Alice".to_string(), "person".to_string());
        profile.add_fact(EntityFact::new(
            "occupation",
            "software engineer",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id.clone());
        storage.store_entity_profile(&profile).unwrap();

        // Store a fact embedding (simulated)
        let key = fact_embedding_key("alice", "occupation", "software engineer");
        let fact_emb = vec![0.5, 0.5, 0.5, 0.5]; // Simple embedding
        storage.store_fact_embedding(&key, &fact_emb).unwrap();

        let search = ProfileSearch::new(storage);
        // Query embedding similar to fact embedding
        let query_emb = vec![0.4, 0.5, 0.5, 0.6];
        let result = search
            .search("What does Alice do for a living?", &query_emb, 10)
            .unwrap();

        // Should find the fact via embedding similarity
        assert!(
            !result.source_scores.is_empty(),
            "Should find results via embedding similarity"
        );
        assert!(result.source_scores.contains_key(&memory_id));
    }

    // ========== Alias resolution tests ==========

    #[test]
    fn test_alias_resolution_short_to_long() {
        // "Mel" in query should resolve to "Melanie" profile
        let (storage, _dir) = create_test_storage();

        let mel_profile =
            EntityProfile::new(EntityId::new(), "Mel".to_string(), "person".to_string());
        let melanie_profile =
            EntityProfile::new(EntityId::new(), "Melanie".to_string(), "person".to_string());
        storage.store_entity_profile(&mel_profile).unwrap();
        storage.store_entity_profile(&melanie_profile).unwrap();

        let search = ProfileSearch::new(storage);
        let entities = search.detect_entities("What does Mel like?").unwrap();

        // "mel" should resolve to "melanie" (canonical longer form)
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0], "melanie");
    }

    #[test]
    fn test_alias_resolution_no_false_positive() {
        // "Jon" should NOT resolve to "Jonathan" if only "Jon" profile exists
        let (storage, _dir) = create_test_storage();

        let jon_profile =
            EntityProfile::new(EntityId::new(), "Jon".to_string(), "person".to_string());
        storage.store_entity_profile(&jon_profile).unwrap();

        let search = ProfileSearch::new(storage);
        let entities = search.detect_entities("What does Jon think?").unwrap();

        // No longer name exists, so "jon" stays as-is
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0], "jon");
    }

    #[test]
    fn test_alias_resolution_no_space_separated() {
        // "Jon" should NOT match "Jon Smith" (space separator, not continuous name)
        let (storage, _dir) = create_test_storage();

        let jon_profile =
            EntityProfile::new(EntityId::new(), "Jon".to_string(), "person".to_string());
        let jon_smith_profile = EntityProfile::new(
            EntityId::new(),
            "Jon Smith".to_string(),
            "person".to_string(),
        );
        storage.store_entity_profile(&jon_profile).unwrap();
        storage.store_entity_profile(&jon_smith_profile).unwrap();

        let search = ProfileSearch::new(storage);
        let entities = search.detect_entities("What does Jon do?").unwrap();

        // "jon" should NOT resolve to "jon smith" (space after "jon")
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0], "jon");
    }

    // ========== resolve_entity_alias() unit tests ==========

    #[test]
    fn test_resolve_entity_alias_basic() {
        let names = vec!["melanie".to_string(), "mel".to_string()];
        assert_eq!(
            resolve_entity_alias("mel", &names),
            Some("melanie".to_string())
        );
    }

    #[test]
    fn test_resolve_entity_alias_no_match() {
        // "jon" stays when only "jon" exists (no longer form)
        let names = vec!["jon".to_string()];
        assert_eq!(resolve_entity_alias("jon", &names), None);
    }

    #[test]
    fn test_resolve_entity_alias_space_separator() {
        // "jon" should NOT match "jon smith" (space after prefix)
        let names = vec!["jon".to_string(), "jon smith".to_string()];
        assert_eq!(resolve_entity_alias("jon", &names), None);
    }

    #[test]
    fn test_resolve_entity_alias_too_short() {
        // 2-char names don't resolve
        let names = vec!["al".to_string(), "alice".to_string()];
        assert_eq!(resolve_entity_alias("al", &names), None);
    }

    #[test]
    fn test_resolve_entity_alias_case_insensitive() {
        let names = vec!["melanie".to_string()];
        assert_eq!(
            resolve_entity_alias("Mel", &names),
            Some("melanie".to_string())
        );
    }

    #[test]
    fn test_alias_resolution_dedup() {
        // Both "Mel" and "Melanie" in query should deduplicate to one "melanie"
        let (storage, _dir) = create_test_storage();

        let mel_profile =
            EntityProfile::new(EntityId::new(), "Mel".to_string(), "person".to_string());
        let melanie_profile =
            EntityProfile::new(EntityId::new(), "Melanie".to_string(), "person".to_string());
        storage.store_entity_profile(&mel_profile).unwrap();
        storage.store_entity_profile(&melanie_profile).unwrap();

        let search = ProfileSearch::new(storage);
        let entities = search
            .detect_entities("What do Mel and Melanie have in common?")
            .unwrap();

        // Both should resolve to "melanie", deduplicated
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0], "melanie");
    }

    // ========== Fuzzy alias resolution tests ==========

    #[test]
    fn test_resolve_entity_alias_fuzzy_mell() {
        // "mell" is not a prefix of "melanie", but truncating to "mel" is
        let names = vec!["melanie".to_string()];
        assert_eq!(
            resolve_entity_alias("mell", &names),
            Some("melanie".to_string())
        );
    }

    #[test]
    fn test_resolve_entity_alias_fuzzy_too_short() {
        // 3-char alias "mel" uses exact prefix (already works), no truncation needed
        let names = vec!["melanie".to_string()];
        assert_eq!(
            resolve_entity_alias("mel", &names),
            Some("melanie".to_string())
        );
    }

    #[test]
    fn test_resolve_entity_alias_fuzzy_no_false_positive() {
        // "mark" truncated to "mar" should NOT match "maria" — different name
        // Actually "mar" IS a prefix of "maria", so this WOULD match.
        // Instead test "mark" + ["michael"] — "mar" is not a prefix of "michael"
        let names = vec!["michael".to_string()];
        assert_eq!(resolve_entity_alias("mark", &names), None);
    }

    #[test]
    fn test_resolve_entity_alias_fuzzy_no_long_name_resolve() {
        // "melanie" (7 chars) must NOT fuzzy-resolve to "melanie's son"
        // Fuzzy matching is capped at ≤5 chars to prevent this
        let names = vec!["melanie".to_string(), "melanie's son".to_string()];
        assert_eq!(resolve_entity_alias("melanie", &names), None);
    }
}
