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
    ///
    /// Includes standard stop words plus conversation-specific junk:
    /// contractions, exclamations, sentence starters, and filler words
    /// that get false-positive extracted as entities in conversation data.
    fn default_stop_words() -> HashSet<String> {
        vec![
            // Standard stop words
            "The", "A", "An", "And", "Or", "But", "In", "On", "At", "To",
            "For", "Of", "With", "By", "From", "As", "Is", "Was", "Are",
            "Were", "Be", "Been", "Being", "Have", "Has", "Had", "Do",
            "Does", "Did", "Will", "Would", "Should", "Could", "May",
            "Might", "Must", "Can", "This", "That", "These", "Those",
            "It", "Its", "He", "She", "They", "We", "You", "I", "Me",
            "My", "Your", "His", "Her", "Their", "Our",
            // Days and months
            "January", "February", "March", "April", "June",
            "July", "August", "September", "October", "November",
            "December", "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
            // Contractions (common in conversation data)
            "I'm", "I'll", "I've", "I'd", "It's", "That's", "What's",
            "There's", "Here's", "He's", "She's", "They're", "We're",
            "You're", "Don't", "Doesn't", "Didn't", "Won't", "Wouldn't",
            "Can't", "Couldn't", "Shouldn't", "Isn't", "Aren't", "Wasn't",
            "Weren't", "Haven't", "Hasn't", "Hadn't", "Let's",
            // Conversation starters and fillers
            "What", "Where", "When", "Who", "Why", "How", "Which",
            "Wow", "Oh", "Ah", "Hmm", "Huh", "Hey", "Hi", "Hello",
            "Well", "So", "Yeah", "Yes", "No", "Nah", "Nope", "Yep",
            "Sure", "Right", "Ok", "Okay", "Thanks", "Thank",
            "Really", "Actually", "Honestly", "Basically", "Totally",
            "Absolutely", "Definitely", "Exactly", "Please", "Sorry",
            "Just", "Like", "Also", "Maybe", "Probably", "Anyway",
            "Though", "Still", "Now", "Then", "Here", "There",
            // Common verbs/adjectives at sentence start
            "Got", "Get", "Go", "Going", "Went", "Come", "Coming",
            "See", "Saw", "Look", "Looking", "Think", "Thinking",
            "Know", "Knew", "Want", "Wanted", "Need", "Needed",
            "Try", "Trying", "Make", "Making", "Take", "Taking",
            "Tell", "Told", "Keep", "Keeping", "Find", "Found",
            "Give", "Gave", "Good", "Great", "Nice", "Cool",
            "Amazing", "Awesome", "Wonderful", "Fantastic",
            "Interesting", "Beautiful", "Glad", "Happy",
            // Other common false positives
            "Some", "Any", "Every", "Each", "All", "Both", "Many",
            "Much", "More", "Most", "Other", "Another", "Same",
            "Such", "Very", "Too", "Even", "Only", "About",
            "After", "Before", "Between", "During", "Since", "Until",
            "If", "Because", "Although", "While", "However",
            "Never", "Always", "Often", "Sometimes", "Already",
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
        // Filter very short words (1-2 chars) — too noisy for entity extraction
        // Exception: single uppercase letters like "C" in "Building C"
        if word.len() == 1 {
            return !word.chars().next().unwrap().is_uppercase();
        }
        if word.len() == 2 {
            return true;
        }

        // Filter stop words (exact match — stop words stored with Title Case)
        if self.stop_words.contains(word) {
            return true;
        }

        // Also check without trailing apostrophe-s (e.g., "Caroline's" → "Caroline's" already handled,
        // but "It's" variants may have different forms)
        let stripped = word.trim_end_matches("'s").trim_end_matches("'t").trim_end_matches("'d")
            .trim_end_matches("'m").trim_end_matches("'ll").trim_end_matches("'ve")
            .trim_end_matches("'re");
        if stripped != word && self.stop_words.contains(stripped) {
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

            // Split into words, preserving apostrophes for contraction detection
            let words: Vec<String> = sentence
                .split_whitespace()
                .map(|w| {
                    w.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                        .to_string()
                })
                .filter(|w| !w.is_empty())
                .collect();

            if words.is_empty() {
                continue;
            }

            // For sentence-first words, only include if they pass the filter
            // The expanded stop words list now covers conversation junk
            // (contractions, fillers, exclamations, common verbs, etc.)
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

        // Should filter out "The", "It", "Monday"
        assert!(!entities.contains(&"The".to_string()));
        assert!(!entities.contains(&"It".to_string()));
        assert!(!entities.contains(&"Monday".to_string()));
    }

    #[test]
    fn test_simple_extractor_sentence_start() {
        let extractor = SimpleEntityExtractor::new();

        let content = "Alice went to the store. The store sells books.";
        let entities = extractor.extract(content).unwrap();

        // Should extract Alice (passes stop words filter)
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

        let content = "I met Alice, then Bob, and finally Charlie.";
        let entities = extractor.extract(content).unwrap();

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

    #[test]
    fn test_simple_extractor_conversation_junk_filtered() {
        let extractor = SimpleEntityExtractor::new();

        // Typical conversation content that was producing junk entities
        let content = "Wow, that's amazing! I'm so happy for you. What are you going to do next? \
                        Oh, I think it's great. Yeah, totally. Thanks for sharing.";
        let entities = extractor.extract(content).unwrap();

        // All of these should be filtered out by expanded stop words
        assert!(!entities.contains(&"Wow".to_string()));
        assert!(!entities.contains(&"I'm".to_string()));
        assert!(!entities.contains(&"What".to_string()));
        assert!(!entities.contains(&"Oh".to_string()));
        assert!(!entities.contains(&"Yeah".to_string()));
        assert!(!entities.contains(&"Thanks".to_string()));
    }

    #[test]
    fn test_simple_extractor_contractions_filtered() {
        let extractor = SimpleEntityExtractor::new();

        let content = "It's raining. That's nice. I've been thinking. Don't worry about it.";
        let entities = extractor.extract(content).unwrap();

        assert!(!entities.contains(&"It's".to_string()));
        assert!(!entities.contains(&"That's".to_string()));
        assert!(!entities.contains(&"I've".to_string()));
        assert!(!entities.contains(&"Don't".to_string()));
    }
}
