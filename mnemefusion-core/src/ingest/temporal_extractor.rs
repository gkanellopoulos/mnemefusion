//! Temporal expression extraction for content-based temporal matching
//!
//! Extracts temporal references from text content (not timestamps/metadata)
//! to enable human-like temporal memory retrieval.

use regex::Regex;
use std::sync::OnceLock;

/// Temporal expression extracted from content
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalExpression {
    /// Relative time reference (yesterday, last week, etc.)
    Relative(String),
    /// Absolute date reference (June 15th, 2023, etc.)
    Absolute(String),
    /// Time of day reference (morning, evening, etc.)
    TimeOfDay(String),
    /// Generic temporal marker (when, during, etc.)
    Generic(String),
}

impl TemporalExpression {
    /// Get the text content of the expression
    pub fn text(&self) -> &str {
        match self {
            TemporalExpression::Relative(s) => s,
            TemporalExpression::Absolute(s) => s,
            TemporalExpression::TimeOfDay(s) => s,
            TemporalExpression::Generic(s) => s,
        }
    }

    /// Convert to a normalized string for matching
    pub fn to_normalized(&self) -> String {
        self.text().to_lowercase()
    }
}

/// Extractor for temporal expressions in text content
pub struct TemporalExtractor {
    relative_regex: Regex,
    absolute_regex: Regex,
    time_of_day_regex: Regex,
    generic_regex: Regex,
}

impl TemporalExtractor {
    /// Create a new temporal extractor with default patterns
    pub fn new() -> Self {
        Self {
            // Relative time: yesterday, today, tomorrow, last week, next month, etc.
            relative_regex: Regex::new(
                r"(?i)\b(yesterday|today|tomorrow|tonight|last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|next\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:this|that)\s+(?:morning|afternoon|evening|night|week|month|year)|(?:\d+|a|an|one|two|three)\s+(?:days?|weeks?|months?|years?)\s+(?:ago|later|from\s+now))\b"
            ).unwrap(),

            // Absolute dates: June 15th, 2023, Jan 1, May 2024, etc.
            absolute_regex: Regex::new(
                r"(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
            ).unwrap(),

            // Time of day: morning, afternoon, evening, night, noon, midnight
            time_of_day_regex: Regex::new(
                r"(?i)\b(morning|afternoon|evening|night|noon|midnight|dawn|dusk|sunrise|sunset)\b"
            ).unwrap(),

            // Generic temporal markers: when, during, while, after, before
            generic_regex: Regex::new(
                r"(?i)\b(when|during|while|after|before|until|since|from|to)\b"
            ).unwrap(),
        }
    }

    /// Extract all temporal expressions from text
    ///
    /// Returns a vector of unique temporal expressions found in the text.
    /// Expressions are deduplicated (case-insensitive).
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to analyze
    ///
    /// # Returns
    ///
    /// Vector of TemporalExpression found in the text
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::ingest::TemporalExtractor;
    ///
    /// let extractor = TemporalExtractor::new();
    /// let expressions = extractor.extract("We had a meeting yesterday about the project");
    /// assert!(!expressions.is_empty());
    /// ```
    pub fn extract(&self, text: &str) -> Vec<TemporalExpression> {
        let mut expressions = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Extract relative time expressions
        for cap in self.relative_regex.captures_iter(text) {
            if let Some(matched) = cap.get(0) {
                let text = matched.as_str().to_string();
                let normalized = text.to_lowercase();
                if seen.insert(normalized) {
                    expressions.push(TemporalExpression::Relative(text));
                }
            }
        }

        // Extract absolute dates
        for cap in self.absolute_regex.captures_iter(text) {
            if let Some(matched) = cap.get(0) {
                let text = matched.as_str().to_string();
                let normalized = text.to_lowercase();
                if seen.insert(normalized) {
                    expressions.push(TemporalExpression::Absolute(text));
                }
            }
        }

        // Extract time of day references
        for cap in self.time_of_day_regex.captures_iter(text) {
            if let Some(matched) = cap.get(0) {
                let text = matched.as_str().to_string();
                let normalized = text.to_lowercase();
                if seen.insert(normalized) {
                    expressions.push(TemporalExpression::TimeOfDay(text));
                }
            }
        }

        // Extract generic temporal markers (only if no other expressions found)
        // This prevents generic words from dominating specific temporal references
        if expressions.is_empty() {
            for cap in self.generic_regex.captures_iter(text) {
                if let Some(matched) = cap.get(0) {
                    let text = matched.as_str().to_string();
                    let normalized = text.to_lowercase();
                    if seen.insert(normalized) {
                        expressions.push(TemporalExpression::Generic(text));
                    }
                }
            }
        }

        expressions
    }

    /// Calculate overlap score between query and memory temporal expressions
    ///
    /// Returns a score from 0.0 to 1.0 indicating how well the temporal contexts match.
    ///
    /// # Arguments
    ///
    /// * `query_expressions` - Temporal expressions from the query
    /// * `memory_expressions` - Temporal expressions from the memory content
    ///
    /// # Returns
    ///
    /// Score from 0.0 (no match) to 1.0 (perfect match)
    ///
    /// # Scoring Logic
    ///
    /// - Exact match (same type, same text): 1.0
    /// - Partial match (same type, different text): 0.5
    /// - Different types: 0.3
    /// - No overlap: 0.0
    pub fn calculate_overlap(
        &self,
        query_expressions: &[TemporalExpression],
        memory_expressions: &[TemporalExpression],
    ) -> f32 {
        if query_expressions.is_empty() || memory_expressions.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut matches = 0;

        // For each query expression, find best match in memory expressions
        for query_expr in query_expressions {
            let mut best_match_score = 0.0;

            for memory_expr in memory_expressions {
                let score = self.match_expressions(query_expr, memory_expr);
                if score > best_match_score {
                    best_match_score = score;
                }
            }

            if best_match_score > 0.0 {
                total_score += best_match_score;
                matches += 1;
            }
        }

        // Average score across all query expressions
        if matches > 0 {
            total_score / query_expressions.len() as f32
        } else {
            0.0
        }
    }

    /// Match two temporal expressions and return similarity score
    fn match_expressions(&self, expr1: &TemporalExpression, expr2: &TemporalExpression) -> f32 {
        use TemporalExpression::*;

        let normalized1 = expr1.to_normalized();
        let normalized2 = expr2.to_normalized();

        // Exact match (case-insensitive)
        if normalized1 == normalized2 {
            return 1.0;
        }

        // Check if same type
        match (expr1, expr2) {
            (Relative(_), Relative(_)) => {
                // Same type, different text - partial match
                0.5
            }
            (Absolute(_), Absolute(_)) => {
                // Check if dates overlap (e.g., "June" matches "June 15")
                if normalized1.contains(&normalized2) || normalized2.contains(&normalized1) {
                    0.7
                } else {
                    0.5
                }
            }
            (TimeOfDay(_), TimeOfDay(_)) => {
                // Same time of day type - partial match
                0.5
            }
            (Generic(_), Generic(_)) => {
                // Generic markers are weak signals
                0.3
            }
            // Different types - weak signal
            _ => 0.3,
        }
    }
}

impl Default for TemporalExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Get a global temporal extractor instance (singleton pattern)
pub fn get_temporal_extractor() -> &'static TemporalExtractor {
    static INSTANCE: OnceLock<TemporalExtractor> = OnceLock::new();
    INSTANCE.get_or_init(TemporalExtractor::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_relative_time() {
        let extractor = TemporalExtractor::new();

        let text = "We had a meeting yesterday about the project";
        let expressions = extractor.extract(text);

        assert_eq!(expressions.len(), 1);
        assert!(matches!(expressions[0], TemporalExpression::Relative(_)));
        assert_eq!(expressions[0].text().to_lowercase(), "yesterday");
    }

    #[test]
    fn test_extract_absolute_date() {
        let extractor = TemporalExtractor::new();

        let text = "The conference was on June 15th, 2023";
        let expressions = extractor.extract(text);

        assert!(!expressions.is_empty());
        assert!(expressions
            .iter()
            .any(|e| matches!(e, TemporalExpression::Absolute(_))));
    }

    #[test]
    fn test_extract_multiple_expressions() {
        let extractor = TemporalExtractor::new();

        let text = "Yesterday morning we discussed the meeting scheduled for next week";
        let expressions = extractor.extract(text);

        // Should find "yesterday", "morning", "next week"
        assert!(expressions.len() >= 2);
    }

    #[test]
    fn test_extract_time_of_day() {
        let extractor = TemporalExtractor::new();

        let text = "The incident happened in the morning";
        let expressions = extractor.extract(text);

        assert_eq!(expressions.len(), 1);
        assert!(matches!(
            expressions[0],
            TemporalExpression::TimeOfDay(_)
        ));
    }

    #[test]
    fn test_extract_no_temporal_expressions() {
        let extractor = TemporalExtractor::new();

        let text = "This is about machine learning techniques";
        let expressions = extractor.extract(text);

        // No temporal expressions in this text
        assert!(expressions.is_empty());
    }

    #[test]
    fn test_extract_deduplicates() {
        let extractor = TemporalExtractor::new();

        let text = "Yesterday we met, and yesterday we decided";
        let expressions = extractor.extract(text);

        // Should deduplicate "yesterday"
        assert_eq!(expressions.len(), 1);
    }

    #[test]
    fn test_calculate_overlap_exact_match() {
        let extractor = TemporalExtractor::new();

        let query_exprs = vec![TemporalExpression::Relative("yesterday".to_string())];
        let memory_exprs = vec![TemporalExpression::Relative("yesterday".to_string())];

        let score = extractor.calculate_overlap(&query_exprs, &memory_exprs);

        // Exact match should give 1.0
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_overlap_partial_match() {
        let extractor = TemporalExtractor::new();

        let query_exprs = vec![TemporalExpression::Relative("yesterday".to_string())];
        let memory_exprs = vec![TemporalExpression::Relative("last week".to_string())];

        let score = extractor.calculate_overlap(&query_exprs, &memory_exprs);

        // Same type, different text - should give 0.5
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_calculate_overlap_no_match() {
        let extractor = TemporalExtractor::new();

        let query_exprs = vec![TemporalExpression::Relative("yesterday".to_string())];
        let memory_exprs = vec![]; // No temporal expressions in memory

        let score = extractor.calculate_overlap(&query_exprs, &memory_exprs);

        // No match should give 0.0
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_calculate_overlap_empty_query() {
        let extractor = TemporalExtractor::new();

        let query_exprs = vec![]; // No temporal expressions in query
        let memory_exprs = vec![TemporalExpression::Relative("yesterday".to_string())];

        let score = extractor.calculate_overlap(&query_exprs, &memory_exprs);

        // Empty query should give 0.0
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_calculate_overlap_multiple_matches() {
        let extractor = TemporalExtractor::new();

        let query_exprs = vec![
            TemporalExpression::Relative("yesterday".to_string()),
            TemporalExpression::TimeOfDay("morning".to_string()),
        ];
        let memory_exprs = vec![
            TemporalExpression::Relative("yesterday".to_string()),
            TemporalExpression::TimeOfDay("morning".to_string()),
        ];

        let score = extractor.calculate_overlap(&query_exprs, &memory_exprs);

        // Both match exactly - should give 1.0
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_absolute_date_formats() {
        let extractor = TemporalExtractor::new();

        // Test various date formats
        let formats = vec![
            "June 15th, 2023",
            "Jun 15 2023",
            "6/15/2023",
            "2023-06-15",
            "May 2024",
        ];

        for format in formats {
            let expressions = extractor.extract(format);
            assert!(
                !expressions.is_empty(),
                "Should extract date from: {}",
                format
            );
            assert!(
                expressions
                    .iter()
                    .any(|e| matches!(e, TemporalExpression::Absolute(_))),
                "Should detect absolute date in: {}",
                format
            );
        }
    }

    #[test]
    fn test_relative_time_variations() {
        let extractor = TemporalExtractor::new();

        let variations = vec![
            ("last week", true),
            ("next month", true),
            ("2 days ago", true),
            ("three weeks from now", true),
            ("this morning", true),
            ("that evening", true),
        ];

        for (text, should_find) in variations {
            let expressions = extractor.extract(text);
            assert_eq!(
                !expressions.is_empty(),
                should_find,
                "Testing: {}",
                text
            );
        }
    }

    #[test]
    fn test_generic_markers_only_when_no_specific() {
        let extractor = TemporalExtractor::new();

        // Text with specific temporal expressions - generic markers should be ignored
        let text1 = "When we met yesterday, we discussed the project";
        let exprs1 = extractor.extract(text1);
        // Should find "yesterday" but not "when" (generic ignored when specific exists)
        assert!(exprs1
            .iter()
            .any(|e| matches!(e, TemporalExpression::Relative(_))));
        assert!(!exprs1
            .iter()
            .any(|e| matches!(e, TemporalExpression::Generic(_))));

        // Text with only generic markers
        let text2 = "When did this happen?";
        let exprs2 = extractor.extract(text2);
        // Should find "when" (generic used when no specific expressions)
        assert!(exprs2
            .iter()
            .any(|e| matches!(e, TemporalExpression::Generic(_))));
    }
}
