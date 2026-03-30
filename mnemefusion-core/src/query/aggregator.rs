//! Multi-turn aggregation for list and collection queries
//!
//! Addresses the multi-turn aggregation problem in conversational memory.
//! When queries ask for lists or collections (e.g., "What activities does X do?"),
//! the answer is often scattered across multiple conversation turns. This module
//! implements collective relevance scoring to find the best combination of turns
//! that collectively answer the query.

use crate::error::Result;
use crate::query::fusion::FusedResult;
use crate::storage::StorageEngine;
use std::collections::{HashMap, HashSet};

/// Query type classification for aggregation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Single-answer extraction query (e.g., "When did X happen?")
    Extraction,
    /// List/aggregation query requiring multiple turns (e.g., "What activities does X do?")
    Aggregation,
    /// Hypothetical/inference query (e.g., "Would X...?")
    Hypothetical,
}

/// Multi-turn aggregator for collective relevance scoring
pub struct MultiTurnAggregator {
    /// Maximum number of turns to combine
    pub max_combination_size: usize,
    /// Minimum relevance score for a turn to be considered
    pub min_turn_score: f32,
    /// Number of top candidates to consider for combinations
    pub candidate_depth: usize,
}

impl Default for MultiTurnAggregator {
    fn default() -> Self {
        Self {
            max_combination_size: 5,
            min_turn_score: 0.0, // RRF scores are ~0.006 max; fusion already filters via semantic_threshold
            candidate_depth: 20,
        }
    }
}

impl MultiTurnAggregator {
    /// Create a new aggregator with custom parameters
    pub fn new(max_combination_size: usize, min_turn_score: f32, candidate_depth: usize) -> Self {
        Self {
            max_combination_size,
            min_turn_score,
            candidate_depth,
        }
    }

    /// Detect if query is an aggregation query requiring multi-turn results
    pub fn classify_query(&self, query_text: &str) -> QueryType {
        let query_lower = query_text.to_lowercase();

        // Hypothetical/inference patterns
        if Self::is_hypothetical(&query_lower) {
            return QueryType::Hypothetical;
        }

        // Aggregation patterns
        if Self::is_aggregation(&query_lower) {
            return QueryType::Aggregation;
        }

        // Default to extraction
        QueryType::Extraction
    }

    /// Check if query is hypothetical/inference type
    fn is_hypothetical(query: &str) -> bool {
        // Starts with would/could/might/should
        if query.starts_with("would ")
            || query.starts_with("could ")
            || query.starts_with("might ")
            || query.starts_with("should ")
        {
            return true;
        }

        // Contains "likely" or "probably"
        if query.contains("likely") || query.contains("probably") {
            return true;
        }

        false
    }

    /// Check if query is an aggregation/list query
    ///
    /// Uses only structural/syntactic patterns — no domain-specific vocabulary.
    /// Queries that don't match fall through to standard extraction handling
    /// with full entity scoring, which is usually more appropriate.
    pub fn is_aggregation(query: &str) -> bool {
        // 1. Explicit list requests (unambiguous)
        if query.contains("list all")
            || query.contains("show all")
            || query.contains("what are all")
            || query.contains("tell me all")
            || query.contains("name all")
        {
            return true;
        }

        // 2. "What are the..." (definite article implies known collection)
        if query.contains("what are the") {
            return true;
        }

        // 3. "In what ways..." (explicitly asks for multiple)
        if query.starts_with("in what ways") {
            return true;
        }

        // 4. "How many..." (counting implies collection context)
        if query.starts_with("how many") {
            return true;
        }

        false
    }

    /// Apply multi-turn aggregation to fusion results
    ///
    /// For aggregation queries, finds the best combination of turns that collectively
    /// answer the query. For extraction queries, returns results as-is.
    ///
    /// # Arguments
    /// * `query_type` - The classified query type
    /// * `query_text` - Original query text
    /// * `candidates` - Fused results from RRF/reranking
    /// * `storage` - Storage engine for memory access
    /// * `limit` - Final number of results to return
    pub fn aggregate(
        &self,
        query_type: QueryType,
        query_text: &str,
        candidates: Vec<FusedResult>,
        storage: &StorageEngine,
        limit: usize,
    ) -> Result<Vec<FusedResult>> {
        // Only apply aggregation for Aggregation query type
        if query_type != QueryType::Aggregation {
            return Ok(candidates.into_iter().take(limit).collect());
        }

        // Take top candidates for aggregation.
        // Scale candidate pool with limit to avoid starving "How many..." queries
        // when limit is large (e.g., 100).
        let effective_depth = if self.candidate_depth > 0 {
            self.candidate_depth.max(limit * 3)
        } else {
            limit * 3
        };
        let to_aggregate = candidates
            .into_iter()
            .take(effective_depth)
            .filter(|r| r.fused_score >= self.min_turn_score)
            .collect::<Vec<_>>();

        if to_aggregate.is_empty() {
            return Ok(vec![]);
        }

        // Extract query terms for answer coverage analysis
        let query_terms = Self::extract_query_terms(query_text);

        // Find best combination using greedy coverage algorithm
        let best_combination =
            self.greedy_coverage_combination(&to_aggregate, &query_terms, storage, limit)?;

        Ok(best_combination)
    }

    /// Greedy algorithm to find combination with best answer coverage
    ///
    /// For each candidate, compute how many unique answer terms it adds.
    /// Greedily select candidates that maximize coverage while maintaining relevance.
    fn greedy_coverage_combination(
        &self,
        candidates: &[FusedResult],
        query_terms: &HashSet<String>,
        storage: &StorageEngine,
        limit: usize,
    ) -> Result<Vec<FusedResult>> {
        let mut selected = Vec::new();
        let mut covered_terms: HashSet<String> = HashSet::new();
        let mut remaining: Vec<_> = candidates.to_vec();

        // Load memory content for all candidates
        let mut content_map: HashMap<String, String> = HashMap::new();
        for result in candidates {
            if let Some(memory) = storage.get_memory_by_u64(result.id.to_u64())? {
                content_map.insert(result.id.to_string(), memory.content);
            }
        }
        // Greedy selection loop
        while selected.len() < limit && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::MIN;

            // Find candidate that adds most new coverage
            for (idx, candidate) in remaining.iter().enumerate() {
                if let Some(content) = content_map.get(&candidate.id.to_string()) {
                    let content_terms = Self::extract_answer_terms(content, query_terms);
                    let new_terms: HashSet<_> = content_terms.difference(&covered_terms).collect();

                    // Score = (new terms added) * (relevance score) + (base relevance)
                    // This balances coverage with relevance
                    let coverage_bonus = new_terms.len() as f32 * 0.3;
                    let score = coverage_bonus + candidate.fused_score;

                    if score > best_score {
                        best_score = score;
                        best_idx = idx;
                    }
                }
            }

            // Add best candidate to selection
            let best = remaining.remove(best_idx);
            if let Some(content) = content_map.get(&best.id.to_string()) {
                let content_terms = Self::extract_answer_terms(content, query_terms);
                covered_terms.extend(content_terms);
            }

            selected.push(best);

            // Early stopping if no significant new coverage
            if selected.len() > 1 && covered_terms.len() < selected.len() {
                break;
            }
        }

        // Sort selected by original fused_score for consistent ordering
        selected.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(selected)
    }

    /// Extract query-relevant terms from text (lowercase, filtered)
    fn extract_query_terms(text: &str) -> HashSet<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|t| t.len() > 2)
            .filter(|t| !Self::is_stopword(t))
            .map(|t| t.to_string())
            .collect()
    }

    /// Extract potential answer terms from content
    ///
    /// Returns words that could be answers - nouns, verbs, adjectives.
    /// Filters out query terms (we want the answers, not the question words).
    fn extract_answer_terms(content: &str, query_terms: &HashSet<String>) -> HashSet<String> {
        content
            .to_lowercase()
            .split_whitespace()
            .filter(|t| t.len() > 2)
            .filter(|t| !Self::is_stopword(t))
            .filter(|t| !query_terms.contains(*t)) // Exclude query terms
            .map(|t| t.to_string())
            .collect()
    }

    /// Check if word is a stopword
    fn is_stopword(word: &str) -> bool {
        const STOPWORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back",
            "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
            "want", "because", "any", "these", "give", "day", "most", "us", "is", "was", "are",
            "been", "has", "had", "were", "did", "does",
        ];

        STOPWORDS.contains(&word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_classification() {
        let aggregator = MultiTurnAggregator::default();

        // Aggregation queries (structural patterns only)
        assert_eq!(
            aggregator.classify_query("What are the books on my shelf?"),
            QueryType::Aggregation
        );
        assert_eq!(
            aggregator.classify_query("List all the topics we discussed"),
            QueryType::Aggregation
        );
        assert_eq!(
            aggregator.classify_query("How many pets does she have?"),
            QueryType::Aggregation
        );

        // Hypothetical queries
        assert_eq!(
            aggregator.classify_query("Would Caroline like this book?"),
            QueryType::Hypothetical
        );
        assert_eq!(
            aggregator.classify_query("What would likely happen if X?"),
            QueryType::Hypothetical
        );

        // Extraction queries (domain-specific words no longer trigger aggregation)
        assert_eq!(
            aggregator.classify_query("When did we meet?"),
            QueryType::Extraction
        );
        assert_eq!(
            aggregator.classify_query("Where is the meeting?"),
            QueryType::Extraction
        );
        assert_eq!(
            aggregator.classify_query("What activities does Melanie do?"),
            QueryType::Extraction
        );
        assert_eq!(
            aggregator.classify_query("What does Caroline do to destress?"),
            QueryType::Extraction
        );
    }

    #[test]
    fn test_extract_query_terms() {
        let terms = MultiTurnAggregator::extract_query_terms("What activities does Melanie do?");

        assert!(terms.contains("activities"));
        assert!(terms.contains("melanie"));
        assert!(!terms.contains("what")); // stopword
        assert!(!terms.contains("does")); // stopword
    }

    #[test]
    fn test_extract_answer_terms() {
        let query_terms: HashSet<String> = ["activities", "melanie"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let content = "Melanie enjoys pottery and camping on weekends";
        let answer_terms = MultiTurnAggregator::extract_answer_terms(content, &query_terms);

        assert!(answer_terms.contains("pottery"));
        assert!(answer_terms.contains("camping"));
        assert!(answer_terms.contains("enjoys"));
        assert!(!answer_terms.contains("melanie")); // query term excluded
        assert!(!answer_terms.contains("activities")); // query term excluded
        assert!(!answer_terms.contains("and")); // stopword
    }

    #[test]
    fn test_stopword_filtering() {
        assert!(MultiTurnAggregator::is_stopword("the"));
        assert!(MultiTurnAggregator::is_stopword("and"));
        assert!(!MultiTurnAggregator::is_stopword("pottery"));
        assert!(!MultiTurnAggregator::is_stopword("camping"));
    }

    #[test]
    fn test_is_hypothetical() {
        assert!(MultiTurnAggregator::is_hypothetical("would x happen?"));
        assert!(MultiTurnAggregator::is_hypothetical("could this work?"));
        assert!(MultiTurnAggregator::is_hypothetical(
            "what would likely occur?"
        ));
        assert!(!MultiTurnAggregator::is_hypothetical("what happened?"));
    }

    #[test]
    fn test_is_aggregation() {
        // Structural patterns that DO trigger aggregation
        assert!(MultiTurnAggregator::is_aggregation("what are the books?"));
        assert!(MultiTurnAggregator::is_aggregation("list all topics"));
        assert!(MultiTurnAggregator::is_aggregation("show all entries"));
        assert!(MultiTurnAggregator::is_aggregation("name all the members"));
        assert!(MultiTurnAggregator::is_aggregation(
            "how many times did she visit?"
        ));
        assert!(MultiTurnAggregator::is_aggregation(
            "in what ways does he help?"
        ));

        // Non-aggregation queries
        assert!(!MultiTurnAggregator::is_aggregation(
            "when did this happen?"
        ));
        assert!(!MultiTurnAggregator::is_aggregation("where is it?"));
        assert!(!MultiTurnAggregator::is_aggregation(
            "what does caroline do to destress?"
        ));
        assert!(!MultiTurnAggregator::is_aggregation(
            "what has she been reading?"
        ));
    }

    #[test]
    fn test_is_aggregation_no_domain_words() {
        // Domain-specific vocabulary should NOT trigger aggregation on its own
        assert!(!MultiTurnAggregator::is_aggregation(
            "what activities does melanie do?"
        ));
        assert!(!MultiTurnAggregator::is_aggregation(
            "does she have any hobbies?"
        ));
        assert!(!MultiTurnAggregator::is_aggregation(
            "what instruments does he play?"
        ));
        assert!(!MultiTurnAggregator::is_aggregation(
            "tell me about her paintings"
        ));
        assert!(!MultiTurnAggregator::is_aggregation(
            "what books has she read?"
        ));
    }

    #[test]
    fn test_aggregate_extraction_respects_limit() {
        // Non-aggregation queries should return up to `limit` results
        use crate::types::MemoryId;

        let aggregator = MultiTurnAggregator::default();

        let candidates: Vec<FusedResult> = (0..50)
            .map(|i| FusedResult {
                id: MemoryId::from_u64(i as u64),
                semantic_score: 0.0,
                bm25_score: 0.0,
                temporal_score: 0.0,
                causal_score: 0.0,
                entity_score: 2.0,
                fused_score: 0.01 - (i as f32 * 0.0001),
                confidence: 0.7,
            })
            .collect();

        let result = aggregator
            .aggregate(
                QueryType::Extraction,
                "when did we meet?",
                candidates,
                &crate::storage::StorageEngine::open(
                    &std::env::temp_dir().join("test_agg_ext.mfdb"),
                )
                .unwrap(),
                25,
            )
            .unwrap();

        assert_eq!(
            result.len(),
            25,
            "Extraction should return exactly limit results"
        );
    }
}
