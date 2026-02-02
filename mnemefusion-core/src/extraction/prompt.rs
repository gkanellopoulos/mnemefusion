//! Optimized prompts for Qwen 3.5 entity extraction
//!
//! These prompts are designed to work with grammar-constrained decoding
//! to produce reliable, structured entity extraction output.

/// Build the extraction prompt for a given text content
///
/// Uses ChatML format compatible with Qwen models.
pub fn build_extraction_prompt(content: &str) -> String {
    // Truncate content if too long to fit in context
    let max_content_len = 1500;
    let truncated_content = if content.len() > max_content_len {
        format!("{}...", &content[..max_content_len])
    } else {
        content.to_string()
    };

    format!(
        r#"<|im_start|>system
You are an entity extraction system. Extract entities and facts from the text.
Output valid JSON only. Be precise and extract only explicitly stated facts.<|im_end|>
<|im_start|>user
Extract entities and facts from this text:

"{content}"

Output JSON with:
- entities: list of {{name, type}} for people/places/organizations mentioned
- entity_facts: list of {{entity, fact_type, value, confidence}} for specific facts
- topics: main subjects discussed
- importance: 0.0-1.0 score

fact_type must be one of: occupation, research_topic, goal, preference, location, relationship, interest, affiliation, characteristic, action<|im_end|>
<|im_start|>assistant
"#,
        content = truncated_content
    )
}

/// Build a few-shot prompt with example for better extraction quality
pub fn build_fewshot_extraction_prompt(content: &str) -> String {
    // Truncate content if too long
    let max_content_len = 1200; // Shorter to leave room for examples
    let truncated_content = if content.len() > max_content_len {
        format!("{}...", &content[..max_content_len])
    } else {
        content.to_string()
    };

    format!(
        r#"<|im_start|>system
You are an entity extraction system. Extract entities and facts from text into structured JSON.<|im_end|>
<|im_start|>user
Extract entities and facts from: "Alice works as a software engineer at Google and lives in Seattle."<|im_end|>
<|im_start|>assistant
{{"entities":[{{"name":"Alice","type":"person"}},{{"name":"Google","type":"organization"}},{{"name":"Seattle","type":"location"}}],"entity_facts":[{{"entity":"Alice","fact_type":"occupation","value":"software engineer","confidence":0.95}},{{"entity":"Alice","fact_type":"affiliation","value":"Google","confidence":0.95}},{{"entity":"Alice","fact_type":"location","value":"Seattle","confidence":0.90}}],"topics":["employment","technology"],"importance":0.7}}<|im_end|>
<|im_start|>user
Extract entities and facts from: "{content}"<|im_end|>
<|im_start|>assistant
"#,
        content = truncated_content
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_extraction_prompt() {
        let prompt = build_extraction_prompt("Alice is a software engineer");
        assert!(prompt.contains("Alice is a software engineer"));
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_prompt_truncation() {
        let long_content = "a".repeat(2000);
        let prompt = build_extraction_prompt(&long_content);
        assert!(prompt.contains("..."));
        assert!(prompt.len() < 3000); // Should be reasonable size
    }

    #[test]
    fn test_fewshot_prompt() {
        let prompt = build_fewshot_extraction_prompt("Bob works at Microsoft");
        assert!(prompt.contains("Bob works at Microsoft"));
        assert!(prompt.contains("Alice works as a software engineer")); // Example
        assert!(prompt.contains("Google")); // From example
    }
}
