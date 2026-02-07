//! Optimized prompts for Qwen 3.5 entity extraction
//!
//! These prompts are designed to work with grammar-constrained decoding
//! to produce reliable, structured entity extraction output.

/// Build the extraction prompt for a given text content
///
/// Uses ChatML format compatible with Qwen models.
/// When `speaker` is provided, it's included in the prompt so the LLM
/// attributes facts to the correct person (the speaker), not to objects mentioned.
pub fn build_extraction_prompt(content: &str, speaker: Option<&str>) -> String {
    // Truncate content if too long to fit in context
    let max_content_len = 1500;
    let truncated_content = if content.len() > max_content_len {
        format!("{}...", &content[..max_content_len])
    } else {
        content.to_string()
    };

    let speaker_instruction = if let Some(name) = speaker {
        format!(
            "\n\nIMPORTANT: This text was spoken by {name}. Attribute facts to {name} as the subject. \
             For example, if {name} says \"I love hiking\", extract: entity={name}, fact_type=interest, value=hiking.",
            name = name
        )
    } else {
        String::new()
    };

    format!(
        r#"<|im_start|>system
You are an entity extraction system. Extract entities and facts from the text.
Output valid JSON only. Be precise and extract only explicitly stated facts.{speaker_instruction}<|im_end|>
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
        speaker_instruction = speaker_instruction,
        content = truncated_content
    )
}

/// Build a few-shot prompt with example for better extraction quality
///
/// When `speaker` is provided, the prompt instructs the LLM to attribute
/// first-person statements to the speaker. This is critical for conversation
/// data where turns are first-person ("I love hiking") but the speaker's
/// identity is only known from metadata.
pub fn build_fewshot_extraction_prompt(content: &str, speaker: Option<&str>) -> String {
    // Truncate content if too long
    let max_content_len = 1000; // Shorter to leave room for examples
    let truncated_content = if content.len() > max_content_len {
        format!("{}...", &content[..max_content_len])
    } else {
        content.to_string()
    };

    let speaker_rule = if let Some(name) = speaker {
        format!(
            "\n7. SPEAKER CONTEXT: This text was spoken by {name}. First-person references (\"I\", \"my\", \"me\") refer to {name}. \
             Attribute facts about the speaker to \"{name}\" as the entity, NOT to the objects mentioned. \
             Example: if {name} says \"Researching adoption agencies\", extract entity=\"{name}\", fact_type=research_topic, value=\"adoption agencies\".",
            name = name
        )
    } else {
        String::new()
    };

    format!(
        r#"<|im_start|>system
You are an expert entity extraction system. Extract entities and facts from text into valid JSON format.

CRITICAL REQUIREMENTS:
1. Return valid JSON only - MUST be parseable
2. For each entity found, extract 2-5 SPECIFIC facts about it
3. entity_facts MUST NEVER be empty if entities are found - extract facts for every entity
4. fact_type MUST be EXACTLY one of these values (no other values allowed):
   - occupation: job/profession/role
   - research_topic: subject being studied/researched
   - goal: what person wants to achieve
   - preference: likes/dislikes/preferences
   - location: where person lives/works/is from
   - relationship: connections to other people/organizations
   - interest: hobbies/interests/passions
   - affiliation: organization/group membership
   - characteristic: personality traits/qualities
   - action: specific activities/actions done
5. DO NOT use values like "activity", "skill", "event", "background" - use the list above only
6. value must be specific text from the input{speaker_rule}<|im_end|>
<|im_start|>user
Extract entities and facts from: "Alice works as a software engineer at Google and lives in Seattle. She is learning machine learning."<|im_end|>
<|im_start|>assistant
{{"entities":[{{"name":"Alice","type":"person"}},{{"name":"Google","type":"organization"}},{{"name":"Seattle","type":"location"}}],"entity_facts":[{{"entity":"Alice","fact_type":"occupation","value":"software engineer","confidence":0.95}},{{"entity":"Alice","fact_type":"affiliation","value":"Google","confidence":0.95}},{{"entity":"Alice","fact_type":"location","value":"Seattle","confidence":0.90}},{{"entity":"Alice","fact_type":"research_topic","value":"machine learning","confidence":0.90}}],"topics":["employment","technology"],"importance":0.8}}<|im_end|>
<|im_start|>user
Extract entities and facts from: "{content}"<|im_end|>
<|im_start|>assistant
"#,
        speaker_rule = speaker_rule,
        content = truncated_content
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_extraction_prompt() {
        let prompt = build_extraction_prompt("Alice is a software engineer", None);
        assert!(prompt.contains("Alice is a software engineer"));
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_build_extraction_prompt_with_speaker() {
        let prompt = build_extraction_prompt("I love hiking", Some("Caroline"));
        assert!(prompt.contains("I love hiking"));
        assert!(prompt.contains("spoken by Caroline"));
        assert!(prompt.contains("Attribute facts to Caroline"));
    }

    #[test]
    fn test_prompt_truncation() {
        let long_content = "a".repeat(2000);
        let prompt = build_extraction_prompt(&long_content, None);
        assert!(prompt.contains("..."));
        assert!(prompt.len() < 3000); // Should be reasonable size
    }

    #[test]
    fn test_fewshot_prompt() {
        let prompt = build_fewshot_extraction_prompt("Bob works at Microsoft", None);
        assert!(prompt.contains("Bob works at Microsoft"));
        assert!(prompt.contains("Alice works as a software engineer")); // Example
        assert!(prompt.contains("Google")); // From example
    }

    #[test]
    fn test_fewshot_prompt_with_speaker() {
        let prompt = build_fewshot_extraction_prompt("Researching adoption agencies", Some("Caroline"));
        assert!(prompt.contains("Researching adoption agencies"));
        assert!(prompt.contains("SPEAKER CONTEXT"));
        assert!(prompt.contains("spoken by Caroline"));
        assert!(prompt.contains("entity=\"Caroline\""));
    }
}
