//! BM25 keyword search index
//!
//! Implements the BM25 (Best Matching 25) algorithm for keyword-based retrieval.
//! This provides exact term matching to complement semantic vector search.
//!
//! Features:
//! - Porter stemming for morphological normalization (research ↔ researching)
//! - Stop word filtering for common words
//! - Length normalization for fair scoring across document sizes
//!
//! BM25 formula:
//! score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
//!
//! where:
//! - f(qi, D) = term frequency of qi in document D
//! - |D| = document length (number of terms)
//! - avgdl = average document length across corpus
//! - k1 = 1.2 (term frequency saturation parameter)
//! - b = 0.75 (length normalization parameter)
//! - IDF(qi) = inverse document frequency
//!
//! Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"

use crate::error::Result;
use crate::types::MemoryId;
use crate::storage::StorageEngine;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

/// Common English stop words to filter out during indexing
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "the", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "can", "could", "may", "might",
    "must", "shall", "should", "would", "now", "also", "like", "even",
    "because", "been", "being", "before", "after", "above", "below",
    "between", "into", "through", "during", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "about", "did",
    "does", "doing", "don", "down", "up", "your", "you", "we", "our",
    "me", "my", "myself", "him", "his", "her", "she", "i", "am",
];

/// BM25 configuration parameters
#[derive(Debug, Clone)]
pub struct BM25Config {
    /// Term frequency saturation parameter (default: 1.2)
    pub k1: f32,
    /// Length normalization parameter (default: 0.75)
    pub b: f32,
    /// Minimum term length to index (default: 2)
    pub min_term_length: usize,
    /// Maximum term length to index (default: 50)
    pub max_term_length: usize,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_term_length: 2,
            max_term_length: 50,
        }
    }
}

/// Document statistics for BM25 scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocumentStats {
    /// Number of terms in document
    length: usize,
    /// Term frequencies: term → count
    term_freqs: HashMap<String, usize>,
}

/// Inverted index: term → list of (doc_id, term_freq)
type InvertedIndex = HashMap<String, Vec<(MemoryId, usize)>>;

/// Document frequency: term → number of documents containing term
type DocumentFrequency = HashMap<String, usize>;

/// Serializable snapshot of BM25 index state.
/// Uses Vec<(K,V)> instead of HashMap where K = MemoryId because serde_json
/// cannot deserialize newtype-wrapped keys from JSON object strings.
#[derive(Serialize, Deserialize)]
struct Bm25State {
    inverted_index: Vec<(String, Vec<(MemoryId, usize)>)>,
    doc_frequency: Vec<(String, usize)>,
    doc_stats: Vec<(MemoryId, DocumentStats)>,
    num_docs: usize,
    avg_doc_length: f32,
}

/// BM25 search result
#[derive(Debug, Clone)]
pub struct BM25Result {
    pub memory_id: MemoryId,
    pub score: f32,
    /// Matching terms from the query
    pub matching_terms: Vec<String>,
}

/// BM25 keyword search index
pub struct BM25Index {
    config: BM25Config,
    /// Inverted index: term → [(doc_id, term_freq)]
    inverted_index: Arc<RwLock<InvertedIndex>>,
    /// Document frequency: term → doc count
    doc_frequency: Arc<RwLock<DocumentFrequency>>,
    /// Document statistics: doc_id → stats
    doc_stats: Arc<RwLock<HashMap<MemoryId, DocumentStats>>>,
    /// Total number of documents
    num_docs: Arc<RwLock<usize>>,
    /// Average document length
    avg_doc_length: Arc<RwLock<f32>>,
    /// Storage reference for persistence
    storage: Arc<StorageEngine>,
    /// Porter stemmer for morphological normalization
    stemmer: Stemmer,
    /// Stop words set for fast lookup
    stop_words: HashSet<&'static str>,
}

impl BM25Index {
    /// Create a new BM25 index
    pub fn new(storage: Arc<StorageEngine>, config: BM25Config) -> Self {
        Self {
            config,
            inverted_index: Arc::new(RwLock::new(HashMap::new())),
            doc_frequency: Arc::new(RwLock::new(HashMap::new())),
            doc_stats: Arc::new(RwLock::new(HashMap::new())),
            num_docs: Arc::new(RwLock::new(0)),
            avg_doc_length: Arc::new(RwLock::new(0.0)),
            storage,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: STOP_WORDS.iter().copied().collect(),
        }
    }

    /// Tokenize and stem text into normalized terms
    ///
    /// Applies:
    /// 1. Lowercase conversion
    /// 2. Punctuation removal
    /// 3. Stop word filtering
    /// 4. Porter stemming (research → research, researching → research)
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter_map(|token| {
                // Remove punctuation from start/end
                let token = token.trim_matches(|c: char| !c.is_alphanumeric());

                // Filter by length
                if token.len() < self.config.min_term_length
                    || token.len() > self.config.max_term_length
                {
                    return None;
                }

                // Skip stop words
                if self.stop_words.contains(token) {
                    return None;
                }

                // Apply Porter stemming
                let stemmed = self.stemmer.stem(token);
                Some(stemmed.to_string())
            })
            .collect()
    }

    /// Calculate term frequencies for a document
    fn calculate_term_freqs(&self, terms: &[String]) -> HashMap<String, usize> {
        let mut freqs = HashMap::new();
        for term in terms {
            *freqs.entry(term.clone()).or_insert(0) += 1;
        }
        freqs
    }

    /// Add a document to the index
    pub fn add(&self, memory_id: &MemoryId, content: &str) -> Result<()> {
        let terms = self.tokenize(content);
        let term_freqs = self.calculate_term_freqs(&terms);
        let doc_length = terms.len();

        // Update document stats
        {
            let mut doc_stats = self.doc_stats.write().unwrap();

            doc_stats.insert(
                memory_id.clone(),
                DocumentStats {
                    length: doc_length,
                    term_freqs: term_freqs.clone(),
                },
            );
        }

        // Update inverted index and document frequency
        {
            let mut inverted_index = self.inverted_index.write().unwrap();

            let mut doc_frequency = self.doc_frequency.write().unwrap();

            for (term, freq) in term_freqs.iter() {
                // Update inverted index
                inverted_index
                    .entry(term.clone())
                    .or_insert_with(Vec::new)
                    .push((memory_id.clone(), *freq));

                // Update document frequency (only count once per document)
                *doc_frequency.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Update total document count and average length
        {
            let mut num_docs = self.num_docs.write().unwrap();
            *num_docs += 1;

            // Recalculate average document length
            let doc_stats = self.doc_stats.read().unwrap();

            let total_length: usize = doc_stats.values().map(|s| s.length).sum();
            let mut avg_doc_length = self.avg_doc_length.write().unwrap();
            *avg_doc_length = total_length as f32 / *num_docs as f32;
        }

        Ok(())
    }

    /// Calculate IDF (Inverse Document Frequency) for a term
    fn calculate_idf(&self, term: &str, num_docs: usize, doc_frequency: &DocumentFrequency) -> f32 {
        let df = doc_frequency.get(term).copied().unwrap_or(0) as f32;
        let n = num_docs as f32;

        // BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Calculate BM25 score for a document given query terms
    fn calculate_bm25_score(
        &self,
        query_terms: &[String],
        doc_stats: &DocumentStats,
        num_docs: usize,
        avg_doc_length: f32,
        doc_frequency: &DocumentFrequency,
    ) -> f32 {
        let k1 = self.config.k1;
        let b = self.config.b;
        let doc_length = doc_stats.length as f32;

        let mut score = 0.0;

        for query_term in query_terms {
            // Get term frequency in document
            let tf = doc_stats.term_freqs.get(query_term).copied().unwrap_or(0) as f32;

            if tf == 0.0 {
                continue; // Term not in document
            }

            // Calculate IDF
            let idf = self.calculate_idf(query_term, num_docs, doc_frequency);

            // BM25 formula
            let numerator = tf * (k1 + 1.0);
            let denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length));

            score += idf * (numerator / denominator);
        }

        score
    }

    /// Search the index with query text
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<BM25Result>> {
        let query_terms = self.tokenize(query);

        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        // Get read locks
        let inverted_index = self.inverted_index.read().unwrap();

        let doc_frequency = self.doc_frequency.read().unwrap();

        let doc_stats = self.doc_stats.read().unwrap();

        let num_docs = *self.num_docs.read().unwrap();

        let avg_doc_length = *self.avg_doc_length.read().unwrap();

        if num_docs == 0 {
            return Ok(Vec::new());
        }

        // Collect candidate documents (documents containing at least one query term)
        let mut candidates: HashMap<MemoryId, Vec<String>> = HashMap::new();

        for query_term in &query_terms {
            if let Some(postings) = inverted_index.get(query_term) {
                for (doc_id, _tf) in postings {
                    candidates
                        .entry(doc_id.clone())
                        .or_insert_with(Vec::new)
                        .push(query_term.clone());
                }
            }
        }

        // Calculate BM25 scores for candidates
        let mut results: Vec<BM25Result> = candidates
            .into_iter()
            .filter_map(|(doc_id, matching_terms)| {
                doc_stats.get(&doc_id).map(|stats| {
                    let score = self.calculate_bm25_score(
                        &query_terms,
                        stats,
                        num_docs,
                        avg_doc_length,
                        &doc_frequency,
                    );

                    BM25Result {
                        memory_id: doc_id,
                        score,
                        matching_terms,
                    }
                })
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k results
        results.truncate(limit);

        Ok(results)
    }

    /// Remove a document from the index
    pub fn remove(&self, memory_id: &MemoryId) -> Result<()> {
        // Get the document's terms before removing
        let terms = {
            let doc_stats = self.doc_stats.read().unwrap();

            doc_stats
                .get(memory_id)
                .map(|stats| stats.term_freqs.keys().cloned().collect::<Vec<_>>())
        };

        if let Some(terms) = terms {
            // Remove from inverted index
            {
                let mut inverted_index = self.inverted_index.write().unwrap();

                let mut doc_frequency = self.doc_frequency.write().unwrap();

                for term in &terms {
                    if let Some(postings) = inverted_index.get_mut(term) {
                        postings.retain(|(id, _)| id != memory_id);

                        if postings.is_empty() {
                            inverted_index.remove(term);
                            doc_frequency.remove(term);
                        } else {
                            // Decrement document frequency
                            if let Some(df) = doc_frequency.get_mut(term) {
                                *df = df.saturating_sub(1);
                            }
                        }
                    }
                }
            }

            // Remove from document stats
            {
                let mut doc_stats = self.doc_stats.write().unwrap();
                doc_stats.remove(memory_id);
            }

            // Update document count and average length
            {
                let mut num_docs = self.num_docs.write().unwrap();
                *num_docs = num_docs.saturating_sub(1);

                if *num_docs > 0 {
                    let doc_stats = self.doc_stats.read().unwrap();

                    let total_length: usize = doc_stats.values().map(|s| s.length).sum();
                    let mut avg_doc_length = self.avg_doc_length.write().unwrap();
                    *avg_doc_length = total_length as f32 / *num_docs as f32;
                } else {
                    let mut avg_doc_length = self.avg_doc_length.write().unwrap();
                    *avg_doc_length = 0.0;
                }
            }
        }

        Ok(())
    }

    /// Get the number of indexed documents
    pub fn num_docs(&self) -> usize {
        *self.num_docs.read().unwrap()
    }

    /// Get the average document length
    pub fn avg_doc_length(&self) -> f32 {
        *self.avg_doc_length.read().unwrap()
    }

    /// Persist BM25 index state to the storage layer.
    ///
    /// Serializes the inverted index, document frequencies, and document
    /// statistics to redb via the METADATA_TABLE. Call after mutations to
    /// ensure durability across restarts.
    pub fn save(&self) -> Result<()> {
        let state = Bm25State {
            inverted_index: self
                .inverted_index
                .read()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            doc_frequency: self
                .doc_frequency
                .read()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            doc_stats: self
                .doc_stats
                .read()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            num_docs: *self.num_docs.read().unwrap(),
            avg_doc_length: *self.avg_doc_length.read().unwrap(),
        };

        let data =
            serde_json::to_vec(&state).map_err(|e| crate::Error::Serialization(e.to_string()))?;
        self.storage.store_bm25_index(&data)
    }

    /// Load BM25 index state from the storage layer.
    ///
    /// If no previously saved state exists, the index remains empty (as
    /// constructed). Call once after `new()` to restore a persisted index.
    pub fn load(&self) -> Result<()> {
        if let Some(data) = self.storage.load_bm25_index()? {
            let state: Bm25State = serde_json::from_slice(&data)
                .map_err(|e| crate::Error::Deserialization(e.to_string()))?;

            *self.inverted_index.write().unwrap() =
                state.inverted_index.into_iter().collect();
            *self.doc_frequency.write().unwrap() =
                state.doc_frequency.into_iter().collect();
            *self.doc_stats.write().unwrap() = state.doc_stats.into_iter().collect();
            *self.num_docs.write().unwrap() = state.num_docs;
            *self.avg_doc_length.write().unwrap() = state.avg_doc_length;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_index() -> (BM25Index, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.mfdb");
        let storage = Arc::new(StorageEngine::open(&db_path).unwrap());
        let index = BM25Index::new(storage, BM25Config::default());
        (index, dir)
    }

    #[test]
    fn test_tokenize() {
        let (index, _dir) = create_test_index();

        // "this", "is", "a" are stop words and get filtered out
        // "hello", "world", "test" remain and get stemmed
        let terms = index.tokenize("Hello, World! This is a test.");
        assert_eq!(terms, vec!["hello", "world", "test"]);

        // Test single letter filtering (min_term_length = 2)
        // "a" and "b" are too short, "cd" and "efg" pass
        let terms = index.tokenize("a b cd efg");
        assert_eq!(terms, vec!["cd", "efg"]);

        // Test stemming: "researching" -> "research", "running" -> "run"
        let terms = index.tokenize("researching running");
        assert_eq!(terms, vec!["research", "run"]);
    }

    #[test]
    fn test_term_frequencies() {
        let (index, _dir) = create_test_index();

        let terms = vec!["hello".to_string(), "world".to_string(), "hello".to_string()];
        let freqs = index.calculate_term_freqs(&terms);

        assert_eq!(freqs.get("hello"), Some(&2));
        assert_eq!(freqs.get("world"), Some(&1));
    }

    #[test]
    fn test_add_and_search() {
        let (index, _dir) = create_test_index();

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let id3 = MemoryId::new();

        index.add(&id1, "The quick brown fox jumps over the lazy dog").unwrap();
        index.add(&id2, "A quick brown dog runs in the park").unwrap();
        index.add(&id3, "The lazy cat sleeps all day").unwrap();

        assert_eq!(index.num_docs(), 3);

        // Search for "quick brown"
        let results = index.search("quick brown", 10).unwrap();
        assert_eq!(results.len(), 2); // id1 and id2 contain both terms
        assert!(results[0].score > 0.0);

        // First result should be id2 (shorter doc with both terms)
        // or id1 depending on BM25 scoring
        assert!(results[0].matching_terms.contains(&"quick".to_string()));
        assert!(results[0].matching_terms.contains(&"brown".to_string()));
    }

    #[test]
    fn test_search_empty_query() {
        let (index, _dir) = create_test_index();

        let id1 = MemoryId::new();
        index.add(&id1, "Test document").unwrap();

        let results = index.search("", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_no_matches() {
        let (index, _dir) = create_test_index();

        let id1 = MemoryId::new();
        index.add(&id1, "The quick brown fox").unwrap();

        let results = index.search("python programming", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_remove() {
        let (index, _dir) = create_test_index();

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        index.add(&id1, "The quick brown fox").unwrap();
        index.add(&id2, "A quick brown dog").unwrap();

        assert_eq!(index.num_docs(), 2);

        // Remove id1
        index.remove(&id1).unwrap();

        assert_eq!(index.num_docs(), 1);

        // Search should only return id2
        let results = index.search("quick brown", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_id, id2);
    }

    #[test]
    fn test_bm25_scoring() {
        let (index, _dir) = create_test_index();

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        // Document with term mentioned once
        index.add(&id1, "The cat sat on the mat").unwrap();

        // Document with term mentioned multiple times (should score higher)
        index.add(&id2, "The cat cat cat sat on the mat mat").unwrap();

        let results = index.search("cat", 10).unwrap();
        assert_eq!(results.len(), 2);

        // id2 should score higher (more term frequency)
        assert_eq!(results[0].memory_id, id2);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_save_and_load_round_trip() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.mfdb");
        let storage = Arc::new(StorageEngine::open(&db_path).unwrap());

        // Populate index
        let index = BM25Index::new(Arc::clone(&storage), BM25Config::default());
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        index
            .add(&id1, "The quick brown fox jumps over the lazy dog")
            .unwrap();
        index
            .add(&id2, "A quick brown dog runs in the park")
            .unwrap();

        // Save
        index.save().unwrap();
        assert_eq!(index.num_docs(), 2);

        // Create fresh index on same storage and load
        let restored = BM25Index::new(Arc::clone(&storage), BM25Config::default());
        assert_eq!(restored.num_docs(), 0); // empty before load
        restored.load().unwrap();
        assert_eq!(restored.num_docs(), 2);

        // Search should return same results
        let results = restored.search("quick brown", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_load_on_empty_db_is_noop() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.mfdb");
        let storage = Arc::new(StorageEngine::open(&db_path).unwrap());

        let index = BM25Index::new(Arc::clone(&storage), BM25Config::default());
        index.load().unwrap(); // no saved state — should not error
        assert_eq!(index.num_docs(), 0);
    }

    #[test]
    fn test_limit() {
        let (index, _dir) = create_test_index();

        for i in 0..10 {
            let id = MemoryId::new();
            index.add(&id, &format!("Document {} with the word test", i)).unwrap();
        }

        let results = index.search("test", 5).unwrap();
        assert_eq!(results.len(), 5);
    }
}
