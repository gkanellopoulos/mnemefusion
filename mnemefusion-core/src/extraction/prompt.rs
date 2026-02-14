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

fact_type must be one of: occupation, research_topic, goal, career_goal, preference, location, relationship, relationship_status, interest, affiliation, characteristic, action, instrument, sport, pet, book, food, hobby, travel, family, event<|im_end|>
<|im_start|>assistant
"#,
        speaker_instruction = speaker_instruction,
        content = truncated_content
    )
}

/// Build a few-shot prompt with examples for entity extraction quality
///
/// Uses two examples: Alice (third-person, general facts) and Bob (third-person,
/// specific biographical facts like instruments, pets, books). The two-shot approach
/// teaches the model both general and specific fact extraction patterns.
///
/// When `speaker` is provided, adds a speaker attribution rule so first-person
/// speech gets correctly attributed to the named speaker.
pub fn build_fewshot_extraction_prompt(content: &str, speaker: Option<&str>) -> String {
    // Truncate content if too long
    let max_content_len = 800; // Shorter to leave room for two examples
    let truncated_content = if content.len() > max_content_len {
        format!("{}...", &content[..max_content_len])
    } else {
        content.to_string()
    };

    let speaker_rule = if let Some(name) = speaker {
        format!(
            "\n\nSPEAKER: This text was spoken by {name}. \
             \"I\"/\"my\"/\"me\" = {name}. Attribute all facts to {name}.",
            name = name
        )
    } else {
        String::new()
    };

    // Example 1: Alice — general facts (occupation, affiliation, location)
    let example1_input = "Alice works as a software engineer at Google and lives in Seattle. She is learning machine learning.";
    let example1_output = "{{\"entities\":[{{\"name\":\"Alice\",\"type\":\"person\"}},{{\"name\":\"Google\",\"type\":\"organization\"}},{{\"name\":\"Seattle\",\"type\":\"location\"}}],\"entity_facts\":[{{\"entity\":\"Alice\",\"fact_type\":\"occupation\",\"value\":\"software engineer\",\"confidence\":0.95}},{{\"entity\":\"Alice\",\"fact_type\":\"affiliation\",\"value\":\"Google\",\"confidence\":0.95}},{{\"entity\":\"Alice\",\"fact_type\":\"location\",\"value\":\"Seattle\",\"confidence\":0.90}},{{\"entity\":\"Alice\",\"fact_type\":\"research_topic\",\"value\":\"machine learning\",\"confidence\":0.90}}],\"topics\":[\"employment\",\"technology\"],\"importance\":0.8}}";

    // Example 2: Bob — specific biographical facts (instrument, pet, book, food)
    let example2_input = "Bob plays guitar and has a hamster named Squeaky. He just finished reading The Hobbit and loves cooking Italian food.";
    let example2_output = "{{\"entities\":[{{\"name\":\"Bob\",\"type\":\"person\"}}],\"entity_facts\":[{{\"entity\":\"Bob\",\"fact_type\":\"instrument\",\"value\":\"guitar\",\"confidence\":0.95}},{{\"entity\":\"Bob\",\"fact_type\":\"pet\",\"value\":\"hamster named Squeaky\",\"confidence\":0.95}},{{\"entity\":\"Bob\",\"fact_type\":\"book\",\"value\":\"The Hobbit\",\"confidence\":0.95}},{{\"entity\":\"Bob\",\"fact_type\":\"food\",\"value\":\"Italian food\",\"confidence\":0.90}}],\"topics\":[\"music\",\"pets\",\"reading\",\"cooking\"],\"importance\":0.7}}";

    // Example 3: Charlie — family, relationship status, career goal, event
    let example3_input = "Charlie is single and has 3 kids who love dinosaurs. He goes running to destress and took his family camping last weekend.";
    let example3_output = "{{\"entities\":[{{\"name\":\"Charlie\",\"type\":\"person\"}}],\"entity_facts\":[{{\"entity\":\"Charlie\",\"fact_type\":\"relationship_status\",\"value\":\"single\",\"confidence\":0.95}},{{\"entity\":\"Charlie\",\"fact_type\":\"family\",\"value\":\"3 children\",\"confidence\":0.95}},{{\"entity\":\"Charlie\",\"fact_type\":\"interest\",\"value\":\"dinosaurs (children)\",\"confidence\":0.80}},{{\"entity\":\"Charlie\",\"fact_type\":\"hobby\",\"value\":\"running\",\"confidence\":0.90}},{{\"entity\":\"Charlie\",\"fact_type\":\"event\",\"value\":\"family camping trip\",\"confidence\":0.85}}],\"topics\":[\"family\",\"hobbies\",\"outdoors\"],\"importance\":0.7}}";

    format!(
        r#"<|im_start|>system
You are a biographical fact extraction specialist. Extract SPECIFIC biographical details from text into valid JSON.

ALWAYS use the MOST SPECIFIC fact_type available:
- instrument: musical instruments played (guitar, piano, violin, clarinet)
- pet: pets owned (include name if given, e.g. "dog named Max")
- book: books read/recommended (include title)
- sport: sports played/watched
- food: food preferences/cooking
- hobby: recreational activities (painting, pottery, running)
- travel: places visited/traveled to
- relationship_status: dating/marital status (single, married, divorced, engaged)
- relationship: connections to specific people (friend, sibling, colleague)
- family: family members and counts (e.g. "3 children", "sister named Amy")
- event: specific events attended or activities done (camping trip, concert, museum visit)
- career_goal: career aspirations or career path chosen
- location: where someone lives/is from/moved from
- occupation: job/profession
- affiliation: organization membership
- goal: general objectives/plans (ONLY if career_goal doesn't fit)
- research_topic: topics being studied
- preference: likes/dislikes
- interest: general interests (ONLY if no specific type above fits)
- characteristic: personality traits
- action: specific actions done (ONLY if no specific type above fits)

WRONG: "I play guitar" -> interest "playing guitar"
RIGHT: "I play guitar" -> instrument "guitar"
WRONG: "My dog Buddy is cute" -> preference "dogs"
RIGHT: "My dog Buddy is cute" -> pet "dog named Buddy"
WRONG: "I read The Hobbit" -> interest "reading"
RIGHT: "I read The Hobbit" -> book "The Hobbit"
WRONG: "I'm single" -> relationship "single"
RIGHT: "I'm single" -> relationship_status "single"
WRONG: "I want to be a counselor" -> goal "be a counselor"
RIGHT: "I want to be a counselor" -> career_goal "counseling"
WRONG: "We went camping" -> action "went camping"
RIGHT: "We went camping" -> event "camping trip"
WRONG: "I have 3 kids" -> characteristic "has children"
RIGHT: "I have 3 kids" -> family "3 children"{speaker_rule}<|im_end|>
<|im_start|>user
Extract entities and facts from: "{example1_input}"<|im_end|>
<|im_start|>assistant
{example1_output}<|im_end|>
<|im_start|>user
Extract entities and facts from: "{example2_input}"<|im_end|>
<|im_start|>assistant
{example2_output}<|im_end|>
<|im_start|>user
Extract entities and facts from: "{example3_input}"<|im_end|>
<|im_start|>assistant
{example3_output}<|im_end|>
<|im_start|>user
Extract entities and facts from: "{content}"<|im_end|>
<|im_start|>assistant
"#,
        speaker_rule = speaker_rule,
        example1_input = example1_input,
        example1_output = example1_output,
        example2_input = example2_input,
        example2_output = example2_output,
        example3_input = example3_input,
        example3_output = example3_output,
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
        let prompt = build_fewshot_extraction_prompt("Dave works at Microsoft", None);
        assert!(prompt.contains("Dave works at Microsoft"));
        // Three examples: Alice (general), Bob (specific), Charlie (family/status)
        assert!(prompt.contains("Alice works as a software engineer"));
        assert!(prompt.contains("Bob plays guitar"));
        assert!(prompt.contains("Charlie is single"));
        // Contrastive WRONG/RIGHT examples in system message
        assert!(prompt.contains("WRONG"));
        assert!(prompt.contains("RIGHT"));
        // Specific fact types demonstrated
        assert!(prompt.contains("instrument"));
        assert!(prompt.contains("pet"));
        assert!(prompt.contains("book"));
        // New fact types
        assert!(prompt.contains("relationship_status"));
        assert!(prompt.contains("family"));
        assert!(prompt.contains("career_goal"));
        assert!(prompt.contains("event"));
    }

    #[test]
    fn test_fewshot_prompt_with_speaker() {
        let prompt = build_fewshot_extraction_prompt("Researching adoption agencies", Some("Caroline"));
        assert!(prompt.contains("Researching adoption agencies"));
        assert!(prompt.contains("spoken by Caroline"));
        assert!(prompt.contains("Attribute all facts to Caroline"));
        // Same two examples for both paths
        assert!(prompt.contains("Alice works as a software engineer"));
        assert!(prompt.contains("Bob plays guitar"));
    }
}
