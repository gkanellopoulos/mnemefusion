//! Causal language pattern extraction for content-based causal matching
//!
//! Detects causal language patterns in text to enable human-like causal reasoning
//! when retrieving memories.

use regex::Regex;
use std::sync::OnceLock;

/// Causal language pattern extractor
pub struct CausalExtractor {
    causal_regex: Regex,
}

impl CausalExtractor {
    /// Create a new causal extractor with default patterns
    pub fn new() -> Self {
        Self {
            // Causal markers: because, caused, led to, resulted in, due to, etc.
            causal_regex: Regex::new(
                r"(?i)\b(because|cause[ds]?|causing|led\s+to|result(?:ed)?\s+(?:in|from)|due\s+to|reason(?:ed)?\s+(?:for|that|why)|therefore|thus|hence|consequently|as\s+a\s+result|why|so\s+that|in\s+order\s+to|since|owing\s+to|thanks\s+to|attributed\s+to|triggered|prompted|enabled|forced|made\s+(?:me|us|them|him|her|it)|explain(?:s|ed)?|explanation)\b"
            ).unwrap(),
        }
    }

    /// Extract causal language patterns from text
    ///
    /// Counts the number of causal markers and calculates causal density.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to analyze
    ///
    /// # Returns
    ///
    /// Tuple of (causal_markers, causal_density)
    /// - causal_markers: List of unique causal marker strings found
    /// - causal_density: Percentage of words that are causal markers (0.0 to 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::ingest::CausalExtractor;
    ///
    /// let extractor = CausalExtractor::new();
    /// let (markers, density) = extractor.extract("The meeting was cancelled because Alice was sick");
    /// assert!(!markers.is_empty());
    /// assert!(density > 0.0);
    /// ```
    pub fn extract(&self, text: &str) -> (Vec<String>, f32) {
        if text.trim().is_empty() {
            return (Vec::new(), 0.0);
        }

        // Count total words
        let word_count = text.split_whitespace().count();
        if word_count == 0 {
            return (Vec::new(), 0.0);
        }

        // Find all causal markers
        let mut causal_markers = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for cap in self.causal_regex.captures_iter(text) {
            if let Some(matched) = cap.get(0) {
                let marker = matched.as_str().to_lowercase();
                if seen.insert(marker.clone()) {
                    causal_markers.push(marker);
                }
            }
        }

        // Calculate causal density
        // Each marker might be multi-word, so count actual words
        let marker_word_count: usize = causal_markers
            .iter()
            .map(|m| m.split_whitespace().count())
            .sum();

        let causal_density = if word_count > 0 {
            marker_word_count as f32 / word_count as f32
        } else {
            0.0
        };

        (causal_markers, causal_density)
    }

    /// Check if a query has causal intent
    ///
    /// Returns true if the query contains causal markers or question words
    /// that indicate causal reasoning (why, how, what caused, etc.)
    ///
    /// # Arguments
    ///
    /// * `query_text` - The query text to analyze
    ///
    /// # Returns
    ///
    /// true if query has causal intent, false otherwise
    pub fn has_causal_intent(&self, query_text: &str) -> bool {
        // Check for causal question patterns
        let query_lower = query_text.to_lowercase();

        // Direct causal questions
        if query_lower.starts_with("why ")
            || query_lower.starts_with("how ")
            || query_lower.contains("what caused")
            || query_lower.contains("what led to")
            || query_lower.contains("reason for")
            || query_lower.contains("because of")
            || query_lower.contains("due to")
            || query_lower.contains("resulted in")
        {
            return true;
        }

        // Check for causal markers in the query
        let (markers, _) = self.extract(query_text);
        !markers.is_empty()
    }

    /// Calculate causal relevance score for a memory
    ///
    /// Scores a memory based on its causal language density, with optional
    /// boost from causal graph connections.
    ///
    /// # Arguments
    ///
    /// * `causal_density` - The causal density of the memory (0.0 to 1.0)
    /// * `has_graph_links` - Whether the memory has causal graph connections
    ///
    /// # Returns
    ///
    /// Score from 0.0 to 1.0
    ///
    /// # Scoring Logic
    ///
    /// - Base score: causal_density * 5.0 (capped at 1.0)
    /// - With graph links: base_score * 1.2 (20% boost, capped at 1.0)
    pub fn calculate_relevance_score(&self, causal_density: f32, has_graph_links: bool) -> f32 {
        // Base score from causal density (multiply by 5 to amplify signal)
        let base_score = (causal_density * 5.0).min(1.0);

        if has_graph_links && base_score > 0.0 {
            // Boost score if memory has causal graph connections
            (base_score * 1.2).min(1.0)
        } else {
            base_score
        }
    }
}

impl Default for CausalExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Get a global causal extractor instance (singleton pattern)
pub fn get_causal_extractor() -> &'static CausalExtractor {
    static INSTANCE: OnceLock<CausalExtractor> = OnceLock::new();
    INSTANCE.get_or_init(CausalExtractor::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_basic_causal() {
        let extractor = CausalExtractor::new();

        let text = "The meeting was cancelled because Alice was sick";
        let (markers, density) = extractor.extract(text);

        assert!(!markers.is_empty());
        assert!(markers.contains(&"because".to_string()));
        assert!(density > 0.0);
    }

    #[test]
    fn test_extract_multiple_causal_markers() {
        let extractor = CausalExtractor::new();

        let text = "The bug was caused by a race condition which led to crashes";
        let (markers, density) = extractor.extract(text);

        // Should find "caused" and "led to"
        assert!(markers.len() >= 2);
        assert!(density > 0.0);
    }

    #[test]
    fn test_extract_no_causal_language() {
        let extractor = CausalExtractor::new();

        let text = "We had a nice lunch today";
        let (markers, density) = extractor.extract(text);

        assert_eq!(markers.len(), 0);
        assert_eq!(density, 0.0);
    }

    #[test]
    fn test_causal_density_calculation() {
        let extractor = CausalExtractor::new();

        // "because" is 1 word out of 8 total = 12.5% density
        let text = "The meeting was cancelled because Alice was sick";
        let (_, density) = extractor.extract(text);

        // Density should be around 0.125 (1/8)
        assert!(density > 0.10 && density < 0.15);
    }

    #[test]
    fn test_causal_density_high() {
        let extractor = CausalExtractor::new();

        // Multiple causal markers in short text = high density
        let text = "Because of this reason, it caused the problem";
        let (_, density) = extractor.extract(text);

        // Should have higher density (multiple markers)
        assert!(density > 0.15);
    }

    #[test]
    fn test_extract_case_insensitive() {
        let extractor = CausalExtractor::new();

        let text = "The issue was CAUSED by incorrect config";
        let (markers, _) = extractor.extract(text);

        assert!(markers.contains(&"caused".to_string()));
    }

    #[test]
    fn test_extract_deduplicates() {
        let extractor = CausalExtractor::new();

        let text = "Because of this, because of that, because of everything";
        let (markers, _) = extractor.extract(text);

        // Should deduplicate "because"
        assert_eq!(
            markers.iter().filter(|m| *m == "because").count(),
            1,
            "Should deduplicate repeated markers"
        );
    }

    #[test]
    fn test_has_causal_intent_why_question() {
        let extractor = CausalExtractor::new();

        assert!(extractor.has_causal_intent("Why was the meeting cancelled?"));
        assert!(extractor.has_causal_intent("How did this happen?"));
        assert!(extractor.has_causal_intent("What caused the error?"));
        assert!(extractor.has_causal_intent("What led to the crash?"));
        assert!(extractor.has_causal_intent("What is the reason for the delay?"));
    }

    #[test]
    fn test_has_causal_intent_non_causal() {
        let extractor = CausalExtractor::new();

        assert!(!extractor.has_causal_intent("What is machine learning?"));
        assert!(!extractor.has_causal_intent("Tell me about the project"));
        assert!(!extractor.has_causal_intent("Show me recent memories"));
    }

    #[test]
    fn test_has_causal_intent_causal_marker() {
        let extractor = CausalExtractor::new();

        // Query contains causal marker
        assert!(extractor.has_causal_intent("because of the issue"));
        assert!(extractor.has_causal_intent("due to the problem"));
    }

    #[test]
    fn test_calculate_relevance_score_no_density() {
        let extractor = CausalExtractor::new();

        let score = extractor.calculate_relevance_score(0.0, false);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_calculate_relevance_score_with_density() {
        let extractor = CausalExtractor::new();

        // Low density (0.1) should be amplified
        let score = extractor.calculate_relevance_score(0.1, false);
        assert!(score > 0.0 && score <= 1.0);
        assert!((score - 0.5).abs() < 0.1); // Should be around 0.5 (0.1 * 5)
    }

    #[test]
    fn test_calculate_relevance_score_with_graph_boost() {
        let extractor = CausalExtractor::new();

        let base_score = extractor.calculate_relevance_score(0.1, false);
        let boosted_score = extractor.calculate_relevance_score(0.1, true);

        assert!(boosted_score > base_score);
        assert!((boosted_score / base_score - 1.2).abs() < 0.01); // 20% boost
    }

    #[test]
    fn test_calculate_relevance_score_capped() {
        let extractor = CausalExtractor::new();

        // High density should be capped at 1.0
        let score = extractor.calculate_relevance_score(0.5, false);
        assert_eq!(score, 1.0);

        // With graph boost should still be capped
        let boosted_score = extractor.calculate_relevance_score(0.5, true);
        assert_eq!(boosted_score, 1.0);
    }

    #[test]
    fn test_extract_various_markers() {
        let extractor = CausalExtractor::new();

        let markers_to_test = vec![
            "because",
            "caused",
            "led to",
            "resulted in",
            "due to",
            "therefore",
            "consequently",
            "thanks to",
            "triggered",
            "explain",
        ];

        for marker in markers_to_test {
            let text = format!("The event {} the outcome", marker);
            let (markers, density) = extractor.extract(&text);
            assert!(
                !markers.is_empty(),
                "Should detect causal marker: {}",
                marker
            );
            assert!(
                density > 0.0,
                "Should have non-zero density for: {}",
                marker
            );
        }
    }

    #[test]
    fn test_extract_empty_text() {
        let extractor = CausalExtractor::new();

        let (markers, density) = extractor.extract("");
        assert_eq!(markers.len(), 0);
        assert_eq!(density, 0.0);
    }

    #[test]
    fn test_extract_whitespace_only() {
        let extractor = CausalExtractor::new();

        let (markers, density) = extractor.extract("   \t\n   ");
        assert_eq!(markers.len(), 0);
        assert_eq!(density, 0.0);
    }
}
