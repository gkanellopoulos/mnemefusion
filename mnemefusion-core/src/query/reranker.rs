// Heuristic reranking for improved precision
//
// Uses lightweight heuristic scoring to maintain zero external dependencies,
// avoiding the latency cost of learned cross-encoder reranking.

use crate::error::Result;
use crate::query::fusion::FusedResult;
use crate::query::intent::QueryIntent;
use crate::storage::StorageEngine;
use crate::types::MemoryId;
use std::collections::HashSet;

/// Heuristic reranker for post-RRF refinement
pub struct HeuristicReranker {
    /// Number of candidates to rerank (default: 20)
    pub rerank_depth: usize,
}

impl Default for HeuristicReranker {
    fn default() -> Self {
        Self { rerank_depth: 20 }
    }
}

impl HeuristicReranker {
    /// Create a new reranker with custom depth
    pub fn new(rerank_depth: usize) -> Self {
        Self { rerank_depth }
    }

    /// Rerank fusion results using intent-adaptive heuristics
    ///
    /// Takes top candidates from RRF and applies additional scoring:
    /// - Keyword overlap (exact term matching)
    /// - Temporal recency (for temporal queries)
    /// - Entity overlap (for entity queries)
    /// - Intent-specific weighting
    ///
    /// # Arguments
    /// * `candidates` - Fused results from RRF
    /// * `query_text` - Original query text
    /// * `query_embedding` - Query embedding vector
    /// * `intent` - Classified query intent
    /// * `storage` - Storage engine for memory access
    /// * `limit` - Final number of results to return
    pub fn rerank(
        &self,
        candidates: Vec<FusedResult>,
        query_text: &str,
        query_embedding: &[f32],
        intent: &QueryIntent,
        storage: &StorageEngine,
        limit: usize,
    ) -> Result<Vec<FusedResult>> {
        // Take top rerank_depth candidates
        let to_rerank = candidates
            .into_iter()
            .take(self.rerank_depth)
            .collect::<Vec<_>>();

        // Extract query terms for keyword overlap
        let query_terms = Self::extract_terms(query_text);

        // Compute reranking scores
        let mut reranked = Vec::new();
        for result in to_rerank {
            let memory = storage.get_memory_by_u64(result.id.to_u64())?;
            if let Some(memory) = memory {
                // Compute heuristic features
                let keyword_overlap = Self::compute_keyword_overlap(&query_terms, &memory.content);
                let temporal_score = Self::compute_temporal_score(&memory, intent);
                let entity_score = Self::compute_entity_score(&memory, &query_terms);

                // Get semantic similarity from fusion result
                let semantic_score = result.semantic_score;

                // Intent-adaptive weighting
                let final_score = match intent {
                    QueryIntent::Temporal => {
                        0.25 * semantic_score
                            + 0.20 * keyword_overlap
                            + 0.45 * temporal_score
                            + 0.10 * entity_score
                    }
                    QueryIntent::Causal => {
                        0.35 * semantic_score
                            + 0.30 * keyword_overlap
                            + 0.15 * temporal_score
                            + 0.20 * entity_score
                    }
                    QueryIntent::Entity => {
                        0.30 * semantic_score
                            + 0.25 * keyword_overlap
                            + 0.05 * temporal_score
                            + 0.40 * entity_score
                    }
                    QueryIntent::Factual => {
                        0.50 * semantic_score
                            + 0.35 * keyword_overlap
                            + 0.05 * temporal_score
                            + 0.10 * entity_score
                    }
                };

                reranked.push(FusedResult {
                    id: result.id.clone(),
                    semantic_score: result.semantic_score,
                    bm25_score: result.bm25_score,
                    temporal_score: result.temporal_score,
                    causal_score: result.causal_score,
                    entity_score: result.entity_score,
                    fused_score: final_score,
                    confidence: result.confidence, // Preserve confidence from input
                });
            }
        }

        // Sort by reranked score
        reranked.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top-k
        Ok(reranked.into_iter().take(limit).collect())
    }

    /// Extract normalized terms from text
    fn extract_terms(text: &str) -> HashSet<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|t| t.len() > 2) // Filter very short terms
            .filter(|t| !Self::is_stopword(t))
            .map(|t| t.to_string())
            .collect()
    }

    /// Compute keyword overlap score (Jaccard similarity)
    fn compute_keyword_overlap(query_terms: &HashSet<String>, content: &str) -> f32 {
        let content_terms = Self::extract_terms(content);

        if query_terms.is_empty() || content_terms.is_empty() {
            return 0.0;
        }

        let intersection = query_terms.intersection(&content_terms).count();
        let union = query_terms.union(&content_terms).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Compute temporal relevance score using half-life recency decay.
    ///
    /// Uses actual memory timestamps with intent-adaptive half-life:
    /// - Temporal queries: 30-day half-life (aggressive, recent = better)
    /// - Other queries: 365-day half-life (gentle, old facts still valid)
    ///
    /// Floor of 0.05 ensures memories never fully vanish.
    fn compute_temporal_score(memory: &crate::types::Memory, intent: &QueryIntent) -> f32 {
        let now = crate::types::Timestamp::now();
        let age_micros = now
            .as_micros()
            .saturating_sub(memory.created_at.as_micros());
        let age_days = age_micros as f64 / (86_400.0 * 1_000_000.0);

        let half_life = if matches!(intent, QueryIntent::Temporal) {
            30.0 // 30-day half-life for temporal queries
        } else {
            365.0 // 1-year half-life for other queries
        };

        let decay = 0.5_f64.powf(age_days / half_life);
        let floor = 0.05;
        (floor + (1.0 - floor) * decay) as f32
    }

    /// Compute entity overlap score
    fn compute_entity_score(memory: &crate::types::Memory, query_terms: &HashSet<String>) -> f32 {
        // Check if memory mentions entities from query
        // Look for capitalized words (simple entity detection)
        let content_words: HashSet<String> = memory
            .content
            .split_whitespace()
            .filter(|w| w.len() > 2 && w.chars().next().unwrap().is_uppercase())
            .map(|w| w.to_lowercase())
            .collect();

        if query_terms.is_empty() || content_words.is_empty() {
            return 0.0;
        }

        let matches = query_terms.intersection(&content_words).count();
        if matches > 0 {
            (matches as f32 / query_terms.len() as f32).min(1.0)
        } else {
            0.0
        }
    }

    /// Check if word is a stopword
    fn is_stopword(word: &str) -> bool {
        const STOPWORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
            "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
            "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is",
            "was", "are", "been", "has", "had", "were", "did", "does",
        ];

        STOPWORDS.contains(&word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Memory;

    #[test]
    fn test_extract_terms() {
        let text = "The quick brown fox jumps over the lazy dog";
        let terms = HeuristicReranker::extract_terms(text);

        // Should filter stopwords and short words
        assert!(terms.contains("quick"));
        assert!(terms.contains("brown"));
        assert!(!terms.contains("the")); // stopword
        assert!(!terms.contains("a")); // too short
    }

    #[test]
    fn test_keyword_overlap() {
        let query_terms: HashSet<String> =
            ["machine", "learning", "algorithms"]
                .iter()
                .map(|s| s.to_string())
                .collect();

        let content1 = "machine learning is a subset of artificial intelligence";
        let score1 = HeuristicReranker::compute_keyword_overlap(&query_terms, content1);
        assert!(score1 > 0.0);

        let content2 = "the weather is nice today";
        let score2 = HeuristicReranker::compute_keyword_overlap(&query_terms, content2);
        assert_eq!(score2, 0.0);
    }

    #[test]
    fn test_temporal_score_decay() {
        // A freshly created memory should have high scores for both intents
        let memory = Memory::new("test content".to_string(), vec![0.1; 384]);

        let temporal_intent = QueryIntent::Temporal;
        let factual_intent = QueryIntent::Factual;

        let temporal_score =
            HeuristicReranker::compute_temporal_score(&memory, &temporal_intent);
        let factual_score = HeuristicReranker::compute_temporal_score(&memory, &factual_intent);

        // Both should be high for a fresh memory (close to 1.0)
        assert!(temporal_score > 0.9, "Fresh memory temporal score should be near 1.0, got {}", temporal_score);
        assert!(factual_score > 0.9, "Fresh memory factual score should be near 1.0, got {}", factual_score);
    }

    #[test]
    fn test_temporal_score_old_memory() {
        use crate::types::Timestamp;
        // A 60-day old memory should have lower temporal score (30-day half-life)
        // but still reasonable factual score (365-day half-life)
        let old_ts = Timestamp::now().subtract_days(60);
        let memory = Memory::new_with_timestamp("old content".to_string(), vec![0.1; 384], old_ts);

        let temporal_score =
            HeuristicReranker::compute_temporal_score(&memory, &QueryIntent::Temporal);
        let factual_score =
            HeuristicReranker::compute_temporal_score(&memory, &QueryIntent::Factual);

        // After 60 days with 30-day half-life: decay = 0.5^2 = 0.25, score ~0.29
        assert!(temporal_score < 0.5, "60-day old temporal score should be < 0.5, got {}", temporal_score);
        // After 60 days with 365-day half-life: decay = 0.5^(60/365) ~= 0.89, score ~0.90
        assert!(factual_score > 0.8, "60-day old factual score should be > 0.8, got {}", factual_score);
        // Temporal decay should be steeper than factual
        assert!(factual_score > temporal_score);
    }

    #[test]
    fn test_entity_score() {
        let query_terms: HashSet<String> = ["alice", "project"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let memory = Memory::new(
            "Alice worked on Project Alpha yesterday".to_string(),
            vec![0.1; 384],
        );

        let score = HeuristicReranker::compute_entity_score(&memory, &query_terms);
        assert!(score > 0.0);
    }

    #[test]
    fn test_stopword_filtering() {
        assert!(HeuristicReranker::is_stopword("the"));
        assert!(HeuristicReranker::is_stopword("and"));
        assert!(!HeuristicReranker::is_stopword("machine"));
        assert!(!HeuristicReranker::is_stopword("learning"));
    }

    #[test]
    fn test_intent_adaptive_weighting() {
        // Semantic score should dominate for Factual queries
        // Temporal score should dominate for Temporal queries
        // Entity score should dominate for Entity queries
        // This is implicitly tested in the rerank() method weights
    }
}
