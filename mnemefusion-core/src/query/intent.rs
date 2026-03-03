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
    /// Extracted entity name for entity-focused queries
    pub entity_focus: Option<String>,
}

/// Classifies query intent using pattern matching
///
/// Uses regex patterns to identify temporal, causal, and entity queries.
/// Falls back to factual intent for generic searches.
pub struct IntentClassifier {
    temporal_patterns: Vec<Regex>,
    causal_patterns: Vec<Regex>,
    entity_patterns: Vec<Regex>,
    /// Patterns for entity-focused list queries
    entity_list_patterns: Vec<Regex>,
}

impl IntentClassifier {
    /// Create a new intent classifier with default patterns
    ///
    /// **Language Note**: Uses English-only patterns for temporal, causal, and entity queries.
    /// Non-English queries will default to `Factual` intent (pure semantic search).
    pub fn new() -> Self {
        Self {
            temporal_patterns: vec![
                // Basic time references
                Regex::new(r"(?i)\b(yesterday|today|tomorrow)\b").unwrap(),
                Regex::new(r"(?i)\b(recent|recently|latest|newest|oldest|earlier)\b").unwrap(),
                Regex::new(r"(?i)\b(when|since|until|before|after)\b").unwrap(),

                // Week/month/year references
                Regex::new(r"(?i)\b(last\s+week|next\s+week|this\s+week)\b").unwrap(),
                Regex::new(r"(?i)\b(last\s+month|next\s+month|this\s+month)\b").unwrap(),
                Regex::new(r"(?i)\b(last\s+year|next\s+year|this\s+year)\b").unwrap(),

                // Month names
                Regex::new(r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\b").unwrap(),

                // Weekday names
                Regex::new(r"(?i)\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b").unwrap(),

                // Time of day
                Regex::new(r"(?i)\b(this\s+morning|this\s+afternoon|this\s+evening|tonight)\b").unwrap(),
                Regex::new(r"(?i)\b(last\s+night|earlier\s+today)\b").unwrap(),

                // Relative time (X units ago)
                Regex::new(r"(?i)\b(\d+\s+(hours?|days?|weeks?|months?|years?)\s+ago)\b").unwrap(),

                // Past/future ranges
                Regex::new(r"(?i)\b(past\s+(few|couple|several))\b").unwrap(),

                // Query patterns
                Regex::new(r"(?i)^(show|list|get)\s+(recent|latest|newest)").unwrap(),
            ],
            causal_patterns: vec![
                // Causal keywords
                Regex::new(r"(?i)\b(why|because|cause[ds]?|reason|led\s+to|result\s+in)\b").unwrap(),
                Regex::new(r"(?i)\b(what\s+caused|what\s+led\s+to|what.*resulted\s+in)\b").unwrap(),
                Regex::new(r"(?i)\b(consequences?|impacts?|effects?|outcomes?)\b").unwrap(),
                Regex::new(r"(?i)^why\s+").unwrap(),
            ],
            entity_patterns: vec![
                // Entity references (capitalized words, "about X", "with Y")
                Regex::new(r"(?i)\b(about|regarding|concerning|related\s+to)\s+[A-Z]").unwrap(),
                Regex::new(r"(?i)\b(with|involving|mention|mentioning)\s+[A-Z]").unwrap(),
                Regex::new(r"\b[A-Z][a-z]+\b").unwrap(), // Capitalized words (potential entities)
            ],
            entity_list_patterns: vec![
                // "What does X like/enjoy/prefer/want/need"
                Regex::new(r"(?i)^what\s+does\s+(\w+)\s+(like|enjoy|prefer|want|need|love|hate|dislike)").unwrap(),
                // "What are X's hobbies/interests/activities/preferences"
                Regex::new(r"(?i)^what\s+(are|were)\s+(\w+)'?s\s+(hobbies|interests|activities|preferences|habits|routines)").unwrap(),
                // "List all/everything about X"
                Regex::new(r"(?i)^(list|show|get|find)\s+(all|everything)\s+(about|for|regarding)\s+(\w+)").unwrap(),
                // "Tell me about X" / "Tell me everything about X"
                Regex::new(r"(?i)^tell\s+me\s+(everything\s+)?(about|regarding)\s+(\w+)").unwrap(),
                // "What do I know about X" / "What do we know about X"
                Regex::new(r"(?i)^what\s+do\s+(i|we)\s+know\s+about\s+(\w+)").unwrap(),
                // "X's hobbies/activities" (direct possessive)
                Regex::new(r"(?i)^(\w+)'?s\s+(hobbies|interests|activities|preferences|habits)").unwrap(),
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
        let entity_list_matches = self.count_matches(&self.entity_list_patterns, query);

        // Calculate scores based on matches
        if temporal_matches > 0 {
            *scores.get_mut(&QueryIntent::Temporal).unwrap() =
                (temporal_matches as f32 * 0.4).min(1.0);
        }

        if causal_matches > 0 {
            *scores.get_mut(&QueryIntent::Causal).unwrap() = (causal_matches as f32 * 0.5).min(1.0);
        }

        if entity_matches > 0 {
            // Entity patterns are weaker indicators
            *scores.get_mut(&QueryIntent::Entity).unwrap() = (entity_matches as f32 * 0.2).min(0.8);
        }

        // Entity list patterns are strong indicators
        if entity_list_matches > 0 {
            // Boost entity score significantly for entity-focused queries
            let current_score = *scores.get(&QueryIntent::Entity).unwrap();
            *scores.get_mut(&QueryIntent::Entity).unwrap() =
                (current_score + entity_list_matches as f32 * 0.6).min(1.0);
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

        // Extract entity name if this is an entity-focused query
        let entity_focus = if primary_intent == QueryIntent::Entity {
            self.extract_entity_from_query(query)
        } else {
            None
        };

        IntentClassification {
            intent: primary_intent,
            confidence: primary_confidence,
            secondary,
            entity_focus,
        }
    }

    /// Count how many patterns match in the query
    fn count_matches(&self, patterns: &[Regex], query: &str) -> usize {
        patterns.iter().filter(|p| p.is_match(query)).count()
    }

    /// Extract entity name from entity-focused queries
    ///
    /// Detects queries like "What does Alice like?" and extracts "Alice".
    /// Used for entity-based pre-retrieval to fetch ALL memories mentioning the entity.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to extract entity from
    ///
    /// # Returns
    ///
    /// Some(entity_name) if an entity-focused pattern matches, None otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::query::IntentClassifier;
    ///
    /// let classifier = IntentClassifier::new();
    /// let entity = classifier.extract_entity_from_query("What does Alice like?");
    /// assert_eq!(entity, Some("Alice".to_string()));
    /// ```
    pub fn extract_entity_from_query(&self, query: &str) -> Option<String> {
        for pattern in &self.entity_list_patterns {
            if let Some(captures) = pattern.captures(query) {
                // Try different capture groups depending on the pattern
                // Most patterns capture entity in group 1 or later groups
                for i in 1..captures.len() {
                    if let Some(capture) = captures.get(i) {
                        let text = capture.as_str().trim();
                        // Skip common words (what, does, are, etc.) and empty strings
                        if !text.is_empty() && !Self::is_common_word(text) && text.len() > 1 {
                            // Capitalize first letter
                            let mut chars = text.chars();
                            if let Some(first) = chars.next() {
                                let capitalized = first.to_uppercase().collect::<String>() + chars.as_str();
                                return Some(capitalized);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Check if word is a common query word (not an entity)
    fn is_common_word(word: &str) -> bool {
        const COMMON_WORDS: &[&str] = &[
            "what", "does", "do", "are", "were", "is", "was", "the", "a", "an",
            "like", "enjoy", "prefer", "want", "need", "love", "hate", "dislike",
            "all", "everything", "about", "for", "regarding", "list", "show", "get",
            "find", "tell", "me", "i", "we", "know", "hobbies", "interests",
            "activities", "preferences", "habits", "routines", "everything ",
        ];
        COMMON_WORDS.contains(&word.to_lowercase().as_str())
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

    // Entity extraction tests
    #[test]
    fn test_entity_extraction_what_does_pattern() {
        let classifier = IntentClassifier::new();

        let entity = classifier.extract_entity_from_query("What does Alice like?");
        assert_eq!(entity, Some("Alice".to_string()));

        let entity = classifier.extract_entity_from_query("what does bob enjoy doing");
        assert_eq!(entity, Some("Bob".to_string()));

        let entity = classifier.extract_entity_from_query("What does Charlie prefer?");
        assert_eq!(entity, Some("Charlie".to_string()));
    }

    #[test]
    fn test_entity_extraction_possessive_pattern() {
        let classifier = IntentClassifier::new();

        let entity = classifier.extract_entity_from_query("What are Alice's hobbies?");
        assert_eq!(entity, Some("Alice".to_string()));

        let entity = classifier.extract_entity_from_query("what were bob's interests");
        assert_eq!(entity, Some("Bob".to_string()));

        let entity = classifier.extract_entity_from_query("Charlie's activities");
        assert_eq!(entity, Some("Charlie".to_string()));
    }

    #[test]
    fn test_entity_extraction_list_pattern() {
        let classifier = IntentClassifier::new();

        let entity = classifier.extract_entity_from_query("List all about Alice");
        assert_eq!(entity, Some("Alice".to_string()));

        let entity = classifier.extract_entity_from_query("show everything about Project");
        assert_eq!(entity, Some("Project".to_string()));

        let entity = classifier.extract_entity_from_query("Tell me everything about Bob");
        assert_eq!(entity, Some("Bob".to_string()));
    }

    #[test]
    fn test_entity_extraction_know_pattern() {
        let classifier = IntentClassifier::new();

        let entity = classifier.extract_entity_from_query("What do I know about Alice?");
        assert_eq!(entity, Some("Alice".to_string()));

        let entity = classifier.extract_entity_from_query("What do we know about system");
        assert_eq!(entity, Some("System".to_string()));
    }

    #[test]
    fn test_entity_extraction_no_match() {
        let classifier = IntentClassifier::new();

        // Generic queries shouldn't extract entities
        let entity = classifier.extract_entity_from_query("What happened yesterday?");
        assert_eq!(entity, None);

        let entity = classifier.extract_entity_from_query("Why was it cancelled?");
        assert_eq!(entity, None);

        let entity = classifier.extract_entity_from_query("machine learning techniques");
        assert_eq!(entity, None);
    }

    #[test]
    fn test_entity_focus_in_classification() {
        let classifier = IntentClassifier::new();

        // Entity query should have entity_focus populated
        let result = classifier.classify("What does Alice like?");
        assert_eq!(result.intent, QueryIntent::Entity);
        assert_eq!(result.entity_focus, Some("Alice".to_string()));

        // Non-entity query should not have entity_focus
        let result = classifier.classify("What happened yesterday?");
        assert_eq!(result.intent, QueryIntent::Temporal);
        assert_eq!(result.entity_focus, None);
    }
}
