//! Intent classification for queries
//!
//! Classifies natural language queries into different intents to optimize
//! retrieval strategy across dimensions (semantic, temporal, causal, entity).
//!
//! **Language Support**: Intent classification currently uses English-only patterns.
//! For non-English queries, the classifier will default to `Factual` intent,
//! which results in pure semantic search (still functional, just suboptimal fusion weights).

use regex::Regex;
use std::collections::HashMap;

/// Query intent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryIntent {
    /// Temporal queries: "what happened yesterday", "recent memories"
    Temporal,
    /// Causal queries: "why did X happen", "what caused Y"
    Causal,
    /// Entity queries: "memories about Alice", "show me project X"
    Entity,
    /// Factual queries: generic semantic search
    Factual,
}

/// Intent classification result with confidence
#[derive(Debug, Clone)]
pub struct IntentClassification {
    /// The primary intent
    pub intent: QueryIntent,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Secondary intents (if any)
    pub secondary: Vec<(QueryIntent, f32)>,
}

/// Classifies query intent using pattern matching
///
/// Uses regex patterns to identify temporal, causal, and entity queries.
/// Falls back to factual intent for generic searches.
pub struct IntentClassifier {
    temporal_patterns: Vec<Regex>,
    causal_patterns: Vec<Regex>,
    entity_patterns: Vec<Regex>,
}

impl IntentClassifier {
    /// Create a new intent classifier with default patterns
    ///
    /// **Language Note**: Uses English-only patterns for temporal, causal, and entity queries.
    /// Non-English queries will default to `Factual` intent (pure semantic search).
    pub fn new() -> Self {
        Self {
            temporal_patterns: vec![
                // Time references
                Regex::new(r"(?i)\b(yesterday|today|tomorrow|last\s+week|next\s+week)\b").unwrap(),
                Regex::new(r"(?i)\b(recent|latest|newest|oldest|earlier)\b").unwrap(),
                Regex::new(r"(?i)\b(when|since|until|before|after)\b").unwrap(),
                Regex::new(r"(?i)\b(on\s+\w+\s+\d+|in\s+(january|february|march|april|may|june|july|august|september|october|november|december))\b").unwrap(),
                Regex::new(r"(?i)\b(\d+\s+(days?|weeks?|months?|years?)\s+ago)\b").unwrap(),
                Regex::new(r"(?i)^(show|list|get)\s+(recent|latest|newest)").unwrap(),
            ],
            causal_patterns: vec![
                // Causal keywords
                Regex::new(r"(?i)\b(why|because|cause[ds]?|reason|led\s+to|result\s+in)\b").unwrap(),
                Regex::new(r"(?i)\b(what\s+caused|what\s+led\s+to|what.*resulted\s+in)\b").unwrap(),
                Regex::new(r"(?i)\b(consequence|impact|effect|outcome)\b").unwrap(),
                Regex::new(r"(?i)^why\s+").unwrap(),
            ],
            entity_patterns: vec![
                // Entity references (capitalized words, "about X", "with Y")
                Regex::new(r"(?i)\b(about|regarding|concerning|related\s+to)\s+[A-Z]").unwrap(),
                Regex::new(r"(?i)\b(with|involving|mention|mentioning)\s+[A-Z]").unwrap(),
                Regex::new(r"\b[A-Z][a-z]+\b").unwrap(), // Capitalized words (potential entities)
            ],
        }
    }

    /// Classify a query string into intent with confidence
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to classify
    ///
    /// # Returns
    ///
    /// IntentClassification with primary intent, confidence, and secondary intents
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::query::IntentClassifier;
    ///
    /// let classifier = IntentClassifier::new();
    /// let result = classifier.classify("Why was the meeting cancelled?");
    /// // Returns Causal intent with high confidence
    /// ```
    pub fn classify(&self, query: &str) -> IntentClassification {
        let mut scores: HashMap<QueryIntent, f32> = HashMap::new();
        scores.insert(QueryIntent::Temporal, 0.0);
        scores.insert(QueryIntent::Causal, 0.0);
        scores.insert(QueryIntent::Entity, 0.0);
        scores.insert(QueryIntent::Factual, 0.3); // Base score for factual

        // Count pattern matches for each intent
        let temporal_matches = self.count_matches(&self.temporal_patterns, query);
        let causal_matches = self.count_matches(&self.causal_patterns, query);
        let entity_matches = self.count_matches(&self.entity_patterns, query);

        // Calculate scores based on matches
        if temporal_matches > 0 {
            *scores.get_mut(&QueryIntent::Temporal).unwrap() =
                (temporal_matches as f32 * 0.4).min(1.0);
        }

        if causal_matches > 0 {
            *scores.get_mut(&QueryIntent::Causal).unwrap() =
                (causal_matches as f32 * 0.5).min(1.0);
        }

        if entity_matches > 0 {
            // Entity patterns are weaker indicators
            *scores.get_mut(&QueryIntent::Entity).unwrap() =
                (entity_matches as f32 * 0.2).min(0.8);
        }

        // Find primary intent (highest score)
        let mut intent_vec: Vec<_> = scores.iter().collect();
        intent_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let primary_intent = *intent_vec[0].0;
        let primary_confidence = *intent_vec[0].1;

        // Gather secondary intents (score > 0.3 and not primary)
        let mut secondary = Vec::new();
        for (intent, score) in intent_vec.iter().skip(1) {
            if **score > 0.3 {
                secondary.push((**intent, **score));
            }
        }

        IntentClassification {
            intent: primary_intent,
            confidence: primary_confidence,
            secondary,
        }
    }

    /// Count how many patterns match in the query
    fn count_matches(&self, patterns: &[Regex], query: &str) -> usize {
        patterns.iter().filter(|p| p.is_match(query)).count()
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_intent() {
        let classifier = IntentClassifier::new();

        let result = classifier.classify("What happened yesterday?");
        assert_eq!(result.intent, QueryIntent::Temporal);
        assert!(result.confidence > 0.3);

        let result = classifier.classify("Show me recent memories");
        assert_eq!(result.intent, QueryIntent::Temporal);

        let result = classifier.classify("What did I do last week?");
        assert_eq!(result.intent, QueryIntent::Temporal);
    }

    #[test]
    fn test_causal_intent() {
        let classifier = IntentClassifier::new();

        let result = classifier.classify("Why was the meeting cancelled?");
        assert_eq!(result.intent, QueryIntent::Causal);
        assert!(result.confidence > 0.4);

        let result = classifier.classify("What caused the server crash?");
        assert_eq!(result.intent, QueryIntent::Causal);

        let result = classifier.classify("What led to the project delay?");
        assert_eq!(result.intent, QueryIntent::Causal);
    }

    #[test]
    fn test_entity_intent() {
        let classifier = IntentClassifier::new();

        let result = classifier.classify("Show me memories about Alice");
        assert_eq!(result.intent, QueryIntent::Entity);
        assert!(result.confidence > 0.1);

        let result = classifier.classify("What do I know about Project Alpha?");
        assert_eq!(result.intent, QueryIntent::Entity);
    }

    #[test]
    fn test_factual_intent() {
        let classifier = IntentClassifier::new();

        let result = classifier.classify("machine learning techniques");
        assert_eq!(result.intent, QueryIntent::Factual);

        let result = classifier.classify("database optimization");
        assert_eq!(result.intent, QueryIntent::Factual);
    }

    #[test]
    fn test_mixed_intent() {
        let classifier = IntentClassifier::new();

        // Temporal + Entity
        let result = classifier.classify("What did Alice do yesterday?");
        // Should have both temporal and entity signals
        assert!(result.secondary.len() > 0 || result.intent == QueryIntent::Temporal);

        // Causal + Temporal
        let result = classifier.classify("Why did the meeting get cancelled last week?");
        // Should prioritize causal
        assert_eq!(result.intent, QueryIntent::Causal);
    }

    #[test]
    fn test_confidence_scores() {
        let classifier = IntentClassifier::new();

        let result = classifier.classify("Why why why");
        // Multiple causal keywords should increase confidence
        assert!(result.confidence > 0.5);

        let result = classifier.classify("yesterday recent latest");
        // Multiple temporal keywords should increase confidence
        assert_eq!(result.intent, QueryIntent::Temporal);
        assert!(result.confidence > 0.4);
    }
}
