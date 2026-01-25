//! Entity extraction from memory content
//!
//! Provides simple entity extraction based on capitalized words.
//! More sophisticated NER can be added in Phase 2.

use crate::Result;
use std::collections::HashSet;

/// Trait for entity extraction strategies
pub trait EntityExtractor {
    /// Extract entities from text content
    ///
    /// # Arguments
    ///
    /// * `content` - The text to extract entities from
    ///
    /// # Returns
    ///
    /// Vector of extracted entities with their canonical names
    fn extract(&self, content: &str) -> Result<Vec<String>>;
}

/// Simple entity extractor based on capitalized words
///
/// Extracts potential entities by finding:
/// - Capitalized words (not at sentence start)
/// - Multi-word capitalized phrases
///
/// Filters out:
/// - Common stop words
/// - Single letters
/// - Words shorter than 2 characters
pub struct SimpleEntityExtractor {
    /// Common stop words to filter out
    stop_words: HashSet<String>,
}

impl SimpleEntityExtractor {
    /// Create a new simple entity extractor
    pub fn new() -> Self {
        let stop_words = Self::default_stop_words();
        Self { stop_words }
    }

    /// Default list of common English stop words
    fn default_stop_words() -> HashSet<String> {
        vec![
            "The",
            "A",
            "An",
            "And",
            "Or",
            "But",
            "In",
            "On",
            "At",
            "To",
            "For",
            "Of",
            "With",
            "By",
            "From",
            "As",
            "Is",
            "Was",
            "Are",
            "Were",
            "Be",
            "Been",
            "Being",
            "Have",
            "Has",
            "Had",
            "Do",
            "Does",
            "Did",
            "Will",
            "Would",
            "Should",
            "Could",
            "May",
            "Might",
            "Must",
            "Can",
            "This",
            "That",
            "These",
            "Those",
            "It",
            "Its",
            "He",
            "She",
            "They",
            "We",
            "You",
            "I",
            "Me",
            "My",
            "Your",
            "His",
            "Her",
            "Their",
            "Our",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    /// Check if a word is capitalized (starts with uppercase letter)
    fn is_capitalized(word: &str) -> bool {
        word.chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
    }

    /// Check if a word should be filtered out
    fn should_filter(&self, word: &str) -> bool {
        // Allow single uppercase letters (e.g., "Building C")
        if word.len() == 1 {
            return !word.chars().next().unwrap().is_uppercase();
        }

        // Filter stop words (case-insensitive check)
        if self.stop_words.contains(word) {
            return true;
        }

        false
    }

    /// Extract multi-word capitalized phrases
    fn extract_phrases(&self, words: &[String]) -> Vec<String> {
        let mut phrases = Vec::new();
        let mut current_phrase = Vec::new();

        for word in words {
            if Self::is_capitalized(word) && !self.should_filter(word) {
                current_phrase.push(word.clone());
            } else {
                // End of phrase
                if current_phrase.len() > 1 {
                    // Multi-word phrase
                    phrases.push(current_phrase.join(" "));
                } else if current_phrase.len() == 1 {
                    // Single capitalized word
                    phrases.push(current_phrase[0].clone());
                }
                current_phrase.clear();
            }
        }

        // Handle phrase at end of text
        if current_phrase.len() > 1 {
            phrases.push(current_phrase.join(" "));
        } else if current_phrase.len() == 1 {
            phrases.push(current_phrase[0].clone());
        }

        phrases
    }
}

impl Default for SimpleEntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityExtractor for SimpleEntityExtractor {
    fn extract(&self, content: &str) -> Result<Vec<String>> {
        // Split into sentences (simple heuristic)
        let sentences: Vec<&str> = content
            .split(|c| c == '.' || c == '!' || c == '?' || c == '\n')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut entities = HashSet::new();

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            // Split into words
            let words: Vec<String> = sentence
                .split_whitespace()
                .map(|w| {
                    // Remove punctuation from end
                    w.trim_end_matches(|c: char| !c.is_alphanumeric())
                        .to_string()
                })
                .filter(|w| !w.is_empty())
                .collect();

            if words.is_empty() {
                continue;
            }

            // For sentence-first words, only skip if they're stop words
            // Otherwise extract all capitalized words including sentence-first
            let start_idx = if self.should_filter(&words[0]) { 1 } else { 0 };

            if words.len() <= start_idx {
                continue;
            }

            // Extract phrases from words
            let phrases = self.extract_phrases(&words[start_idx..]);

            for phrase in phrases {
                entities.insert(phrase);
            }
        }

        // Convert to vector and sort for determinism
        let mut result: Vec<String> = entities.into_iter().collect();
        result.sort();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_extractor_basic() {
        let extractor = SimpleEntityExtractor::new();

        let content = "I met Alice at the conference. Bob was there too.";
        let entities = extractor.extract(content).unwrap();

        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_simple_extractor_multi_word() {
        let extractor = SimpleEntityExtractor::new();

        let content = "Project Alpha is managed by Team Beta. We work at Acme Corp.";
        let entities = extractor.extract(content).unwrap();

        assert!(entities.contains(&"Project Alpha".to_string()));
        assert!(entities.contains(&"Team Beta".to_string()));
        assert!(entities.contains(&"Acme Corp".to_string()));
    }

    #[test]
    fn test_simple_extractor_filters_stop_words() {
        let extractor = SimpleEntityExtractor::new();

        let content = "The meeting was at Building C. It was on Monday.";
        let entities = extractor.extract(content).unwrap();

        // Should extract Building C
        assert!(entities.contains(&"Building C".to_string()));

        // Should filter out "The", "It"
        assert!(!entities.contains(&"The".to_string()));
        assert!(!entities.contains(&"It".to_string()));
        assert!(!entities.contains(&"Monday".to_string()));
    }

    #[test]
    fn test_simple_extractor_sentence_start() {
        let extractor = SimpleEntityExtractor::new();

        let content = "Alice went to the store. The store sells books.";
        let entities = extractor.extract(content).unwrap();

        // Should extract Alice (not at sentence start in logical sense)
        assert!(entities.contains(&"Alice".to_string()));

        // Should not extract "The" even though it's capitalized at sentence start
        assert!(!entities.contains(&"The".to_string()));
    }

    #[test]
    fn test_simple_extractor_acronyms() {
        let extractor = SimpleEntityExtractor::new();

        let content = "We work at NASA and collaborate with MIT.";
        let entities = extractor.extract(content).unwrap();

        assert!(entities.contains(&"NASA".to_string()));
        assert!(entities.contains(&"MIT".to_string()));
    }

    #[test]
    fn test_simple_extractor_punctuation() {
        let extractor = SimpleEntityExtractor::new();

        // Simple extractor handles basic punctuation
        // Note: Consecutive capitalized words are treated as multi-word entities
        let content = "I met Alice, then Bob, and finally Charlie.";
        let entities = extractor.extract(content).unwrap();

        // Should extract names despite commas
        // Names appear after lowercase words so they're extracted separately
        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"Bob".to_string()));
        assert!(entities.contains(&"Charlie".to_string()));
    }

    #[test]
    fn test_simple_extractor_empty() {
        let extractor = SimpleEntityExtractor::new();

        let content = "nothing here is capitalized except sentence starts.";
        let entities = extractor.extract(content).unwrap();

        assert_eq!(entities.len(), 0);
    }

    #[test]
    fn test_simple_extractor_determinism() {
        let extractor = SimpleEntityExtractor::new();

        let content = "Alice, Bob, and Charlie are working on Project X.";

        let entities1 = extractor.extract(content).unwrap();
        let entities2 = extractor.extract(content).unwrap();

        // Results should be deterministic (sorted)
        assert_eq!(entities1, entities2);
    }
}
