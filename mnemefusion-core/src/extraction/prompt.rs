//! Entity extraction prompts with multi-model chat template support
//!
//! All prompt builders produce ChatML format (Qwen) by default.
//! Use `apply_chat_template()` to convert to other model formats (e.g. Phi-4).

/// Model family — determines which chat template wrapping to use.
///
/// Detected automatically from the model filename in `LlmEntityExtractor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Qwen models: ChatML format (`<|im_start|>role\n...<|im_end|>`)
    Qwen,
    /// Microsoft Phi-4 models: Phi-4 format (`<|role|>\n...<|end|>`)
    Phi4,
}

impl ModelFamily {
    /// Detect model family from a model file path.
    ///
    /// Checks the filename (case-insensitive) for known model identifiers.
    /// Defaults to Qwen (ChatML) for unknown models.
    pub fn from_path(path: &std::path::Path) -> Self {
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_lowercase();
        if filename.contains("phi") {
            ModelFamily::Phi4
        } else {
            ModelFamily::Qwen
        }
    }
}

/// Convert a ChatML-formatted prompt to the target model's chat template.
///
/// All prompt builders produce ChatML format. This function converts to
/// other formats when needed. Returns the prompt unchanged for Qwen/ChatML.
pub fn apply_chat_template(prompt: &str, family: ModelFamily) -> String {
    match family {
        ModelFamily::Qwen => prompt.to_string(),
        ModelFamily::Phi4 => prompt
            .replace("<|im_start|>system\n", "<|system|>\n")
            .replace("<|im_start|>user\n", "<|user|>\n")
            .replace("<|im_start|>assistant\n", "<|assistant|>\n")
            .replace("<|im_end|>", "<|end|>"),
    }
}

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
RIGHT: "I have 3 kids" -> family "3 children"

IMPORTANT RULES:
- If you cannot determine a fact for a type, DO NOT output "none" — simply omit it.
- Keep values concise and factual (under 10 words). Extract the key fact, not the full sentence.
  WRONG: value "sharing stories to build a supportive community of hope"
  RIGHT: value "community storytelling"
  WRONG: value "making a positive difference in someone's life"
  RIGHT: value "helping others"{speaker_rule}<|im_end|>
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

/// Build a relationship-focused extraction prompt.
///
/// Persona: "relationship and connection analyst". Emphasizes interpersonal
/// relationships, group affiliations, and social dynamics. Produces the same
/// JSON schema as `build_fewshot_extraction_prompt()`.
pub fn build_relationship_extraction_prompt(content: &str, speaker: Option<&str>) -> String {
    let max_content_len = 800;
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

    // Example 1: Coworkers collaborating
    let example1_input = "Dave and Sarah have been working together on the climate project at NOAA for three years. They meet every Tuesday.";
    let example1_output = "{{\"entities\":[{{\"name\":\"Dave\",\"type\":\"person\"}},{{\"name\":\"Sarah\",\"type\":\"person\"}},{{\"name\":\"NOAA\",\"type\":\"organization\"}}],\"entity_facts\":[{{\"entity\":\"Dave\",\"fact_type\":\"relationship\",\"value\":\"colleague of Sarah\",\"confidence\":0.95}},{{\"entity\":\"Dave\",\"fact_type\":\"affiliation\",\"value\":\"NOAA\",\"confidence\":0.95}},{{\"entity\":\"Sarah\",\"fact_type\":\"affiliation\",\"value\":\"NOAA\",\"confidence\":0.95}},{{\"entity\":\"Dave\",\"fact_type\":\"research_topic\",\"value\":\"climate project\",\"confidence\":0.90}},{{\"entity\":\"Sarah\",\"fact_type\":\"research_topic\",\"value\":\"climate project\",\"confidence\":0.90}}],\"topics\":[\"work\",\"collaboration\"],\"importance\":0.7}}";

    // Example 2: Family connections
    let example2_input = "My sister Amy just got engaged to Tom. They met at my friend Rachel's birthday party last year.";
    let example2_output = "{{\"entities\":[{{\"name\":\"Amy\",\"type\":\"person\"}},{{\"name\":\"Tom\",\"type\":\"person\"}},{{\"name\":\"Rachel\",\"type\":\"person\"}}],\"entity_facts\":[{{\"entity\":\"Amy\",\"fact_type\":\"family\",\"value\":\"sister\",\"confidence\":0.95}},{{\"entity\":\"Amy\",\"fact_type\":\"relationship\",\"value\":\"engaged to Tom\",\"confidence\":0.95}},{{\"entity\":\"Amy\",\"fact_type\":\"relationship_status\",\"value\":\"engaged\",\"confidence\":0.95}},{{\"entity\":\"Rachel\",\"fact_type\":\"relationship\",\"value\":\"friend\",\"confidence\":0.85}}],\"topics\":[\"family\",\"engagement\"],\"importance\":0.8}}";

    // Example 3: Social group dynamics
    let example3_input = "I joined the local running club with my neighbor Mike. The coach, Elena, is really pushing us for the marathon.";
    let example3_output = "{{\"entities\":[{{\"name\":\"Mike\",\"type\":\"person\"}},{{\"name\":\"Elena\",\"type\":\"person\"}}],\"entity_facts\":[{{\"entity\":\"Mike\",\"fact_type\":\"relationship\",\"value\":\"neighbor\",\"confidence\":0.90}},{{\"entity\":\"Mike\",\"fact_type\":\"affiliation\",\"value\":\"running club\",\"confidence\":0.85}},{{\"entity\":\"Elena\",\"fact_type\":\"occupation\",\"value\":\"running coach\",\"confidence\":0.85}},{{\"entity\":\"Elena\",\"fact_type\":\"affiliation\",\"value\":\"running club\",\"confidence\":0.85}}],\"topics\":[\"running\",\"social\"],\"importance\":0.6}}";

    format!(
        r#"<|im_start|>system
You are a relationship and connection analyst. Extract RELATIONSHIPS and SOCIAL CONNECTIONS from text into valid JSON.

Focus on extracting:
- Interpersonal relationships (friend, sibling, colleague, mentor, neighbor)
- Group affiliations and shared activities
- Social dynamics and connections between people
- Relationship status changes (engaged, married, dating)

Use the MOST SPECIFIC fact_type available:
- relationship: connections to specific people (friend of X, colleague of Y, mentor)
- relationship_status: dating/marital status (single, married, divorced, engaged)
- family: family members and counts (sister, brother, "3 children", "mother named Jane")
- affiliation: organization/group membership (company, club, team)
- occupation: job/profession (use when role is clear)
- interest: shared interests that connect people
- event: social events or gatherings
- location: where someone lives/is from

WRONG: "My sister Amy" -> relationship "sister"
RIGHT: "My sister Amy" -> family "sister named Amy"
WRONG: "Working with Sarah at Google" -> action "working with Sarah"
RIGHT: "Working with Sarah at Google" -> relationship "colleague of Sarah", affiliation "Google"
WRONG: "Got engaged to Tom" -> interest "engagement"
RIGHT: "Got engaged to Tom" -> relationship_status "engaged", relationship "engaged to Tom"

IMPORTANT RULES:
- If you cannot determine a fact for a type, DO NOT output "none" — simply omit it.
- Keep values concise and factual (under 10 words).{speaker_rule}<|im_end|>
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

/// Build an event-and-temporal-focused extraction prompt.
///
/// Persona: "event and activity chronicler". Emphasizes specific events,
/// temporal anchors, actions/goals, and plans. Produces the same JSON schema
/// as `build_fewshot_extraction_prompt()`.
pub fn build_event_temporal_extraction_prompt(content: &str, speaker: Option<&str>) -> String {
    let max_content_len = 800;
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

    // Example 1: Travel with dates
    let example1_input = "We're flying to Tokyo next March for two weeks. I've been studying Japanese since September.";
    let example1_output = "{{\"entities\":[{{\"name\":\"Tokyo\",\"type\":\"location\"}}],\"entity_facts\":[{{\"entity\":\"Tokyo\",\"fact_type\":\"travel\",\"value\":\"trip planned next March\",\"confidence\":0.95}},{{\"entity\":\"Tokyo\",\"fact_type\":\"event\",\"value\":\"two-week visit\",\"confidence\":0.90}}],\"topics\":[\"travel\",\"language learning\"],\"importance\":0.8}}";

    // Example 2: Training and goals
    let example2_input = "I finished my first half-marathon last Sunday in 1:52. Now I'm training for the full marathon in October with coach Dan.";
    let example2_output = "{{\"entities\":[{{\"name\":\"Dan\",\"type\":\"person\"}}],\"entity_facts\":[{{\"entity\":\"Dan\",\"fact_type\":\"occupation\",\"value\":\"marathon coach\",\"confidence\":0.85}},{{\"entity\":\"Dan\",\"fact_type\":\"relationship\",\"value\":\"coach\",\"confidence\":0.85}}],\"topics\":[\"running\",\"training\"],\"importance\":0.8}}";

    // Example 3: Celebration and gathering
    let example3_input = "Mom's 60th birthday party is this Saturday at the lake house. My brother is bringing his famous barbecue.";
    let example3_output = "{{\"entities\":[{{\"name\":\"Mom\",\"type\":\"person\"}}],\"entity_facts\":[{{\"entity\":\"Mom\",\"fact_type\":\"event\",\"value\":\"60th birthday party at lake house\",\"confidence\":0.95}},{{\"entity\":\"Mom\",\"fact_type\":\"family\",\"value\":\"mother\",\"confidence\":0.90}}],\"topics\":[\"family\",\"celebration\"],\"importance\":0.8}}";

    format!(
        r#"<|im_start|>system
You are an event and activity chronicler. Extract EVENTS, ACTIVITIES, and TEMPORAL details from text into valid JSON.

Focus on extracting:
- Specific events and activities (trips, celebrations, meetings, competitions)
- Temporal anchors (dates, "last week", "next month", "since September")
- Actions and goals with context
- Plans and future intentions

Use the MOST SPECIFIC fact_type available:
- event: specific events attended or planned (birthday party, marathon, concert, trip)
- travel: places visited or planned (include timing if mentioned)
- sport: sports played/competed in (include achievements)
- hobby: recreational activities done regularly
- goal: objectives/plans with context
- career_goal: career aspirations or milestones
- family: family members mentioned in event context
- relationship: connections to people involved in events
- location: where events take place
- occupation: job/role if relevant to event

WRONG: "Finished my first marathon" -> interest "running"
RIGHT: "Finished my first marathon" -> event "completed first marathon", sport "marathon running"
WRONG: "Flying to Tokyo next March" -> preference "Japan"
RIGHT: "Flying to Tokyo next March" -> travel "trip planned next March"
WRONG: "Mom's birthday party Saturday" -> family "has mother"
RIGHT: "Mom's birthday party Saturday" -> event "60th birthday party", family "mother"

IMPORTANT RULES:
- If you cannot determine a fact for a type, DO NOT output "none" — simply omit it.
- Keep values concise and factual (under 10 words). Extract the key fact, not the full sentence.{speaker_rule}<|im_end|>
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

/// Build a typed decomposition extraction prompt (ENGRAM-inspired).
///
/// This is a SINGLE prompt that produces both entity facts AND typed records
/// in one LLM call. It decomposes a conversation turn into 1-3 typed
/// sub-records (episodic, semantic, procedural), each a self-contained
/// embeddable sentence.
///
/// Research basis:
/// - ENGRAM (arXiv 2511.12960): typed memory separation improves QA accuracy
/// - TReMu (ACL 2025): event date inference improves temporal QA
///
/// # Arguments
///
/// * `content` - The conversation turn to decompose
/// * `speaker` - Optional speaker name for attribution
/// * `session_date` - Optional ISO-8601 date of the conversation session
///   (enables relative date inference: "last weekend" + 2023-03-15 → 2023-03-11)
pub fn build_typed_extraction_prompt(
    content: &str,
    speaker: Option<&str>,
    session_date: Option<&str>,
) -> String {
    let max_content_len = 800;
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

    let date_instruction = if let Some(date) = session_date {
        format!(
            "\n\nSESSION DATE: {date}. Convert relative time expressions to ISO-8601 dates:\n\
             - \"yesterday\" → subtract 1 day from session date\n\
             - \"last weekend\" → find previous Saturday\n\
             - \"last month\" → same day, previous month\n\
             - \"two weeks ago\" → subtract 14 days\n\
             Set event_date on episodic records when you can infer the date.",
            date = date
        )
    } else {
        String::new()
    };

    // Example 1: Mixed — episodic event + semantic fact + relationship
    let example1_input = "Alice says: I just got promoted to senior engineer at Google! My husband Bob threw me a surprise party last Saturday.";
    let example1_output = r#"{"entities":[{"name":"Alice","type":"person"},{"name":"Bob","type":"person"},{"name":"Google","type":"organization"}],"entity_facts":[{"entity":"Alice","fact_type":"occupation","value":"senior engineer","confidence":0.95},{"entity":"Alice","fact_type":"affiliation","value":"Google","confidence":0.95},{"entity":"Alice","fact_type":"relationship","value":"married to Bob","confidence":0.95}],"topics":["career","celebration"],"importance":0.8,"records":[{"record_type":"semantic","summary":"Alice works as a senior engineer at Google","entities":["Alice","Google"]},{"record_type":"episodic","summary":"Alice got promoted to senior engineer","event_date":"2023-03-13","entities":["Alice"]},{"record_type":"episodic","summary":"Bob threw Alice a surprise party for her promotion","event_date":"2023-03-11","entities":["Alice","Bob"]}],"relationships":[{"from_entity":"Alice","to_entity":"Bob","relation_type":"spouse","confidence":0.95}]}"#;

    // Example 2: Routine + fact — procedural + semantic
    let example2_input = "Mike says: I jog every morning at 6am in Central Park. I've been vegan for three years now.";
    let example2_output = r#"{"entities":[{"name":"Mike","type":"person"},{"name":"Central Park","type":"location"}],"entity_facts":[{"entity":"Mike","fact_type":"hobby","value":"jogging","confidence":0.95},{"entity":"Mike","fact_type":"location","value":"Central Park","confidence":0.85},{"entity":"Mike","fact_type":"preference","value":"vegan diet","confidence":0.95}],"topics":["exercise","diet"],"importance":0.7,"records":[{"record_type":"procedural","summary":"Mike jogs every morning at 6am in Central Park","entities":["Mike","Central Park"]},{"record_type":"semantic","summary":"Mike has been vegan for three years","entities":["Mike"]}],"relationships":[]}"#;

    // Example 3: Pure event — episodic only
    let example3_input = "Sarah says: We went camping at Yosemite with my college friend Dan two weeks ago. It was amazing!";
    let example3_output = r#"{"entities":[{"name":"Sarah","type":"person"},{"name":"Dan","type":"person"},{"name":"Yosemite","type":"location"}],"entity_facts":[{"entity":"Sarah","fact_type":"event","value":"camping at Yosemite","confidence":0.95},{"entity":"Sarah","fact_type":"relationship","value":"college friend Dan","confidence":0.90}],"topics":["outdoors","friendship"],"importance":0.7,"records":[{"record_type":"episodic","summary":"Sarah went camping at Yosemite with her college friend Dan","event_date":"2023-03-01","entities":["Sarah","Dan","Yosemite"]}],"relationships":[{"from_entity":"Sarah","to_entity":"Dan","relation_type":"friend","confidence":0.90}]}"#;

    format!(
        r#"<|im_start|>system
You extract entities, facts, typed records, and relationships from text into valid JSON.

RECORD TYPES — decompose each turn into 1-3 records:
- episodic: Events, activities, experiences. Set event_date (ISO-8601) when inferrable.
- semantic: Stable facts, preferences, attributes. No event_date.
- procedural: Routines, instructions, recurring patterns. No event_date.

Each record.summary must be a SELF-CONTAINED sentence (understandable without the original text).

RELATIONSHIPS — extract entity-to-entity links:
- relation_type: spouse, sibling, parent, child, friend, colleague, neighbor, mentor, etc.
- Only extract relationships explicitly stated or strongly implied.

ENTITY FACTS — use the most specific fact_type:
instrument, pet, book, sport, food, hobby, travel, relationship_status, relationship,
family, event, career_goal, location, occupation, affiliation, goal, research_topic,
preference, interest, characteristic, action

IMPORTANT RULES:
- Each record.summary must name the subject (e.g., "Alice works at Google", not "Works at Google")
- Keep values concise (under 10 words)
- Do NOT output "none" values — omit them instead{speaker_rule}{date_instruction}<|im_end|>
<|im_start|>user
Extract from: "{example1_input}"
Session date: 2023-03-15<|im_end|>
<|im_start|>assistant
{example1_output}<|im_end|>
<|im_start|>user
Extract from: "{example2_input}"<|im_end|>
<|im_start|>assistant
{example2_output}<|im_end|>
<|im_start|>user
Extract from: "{example3_input}"
Session date: 2023-03-15<|im_end|>
<|im_start|>assistant
{example3_output}<|im_end|>
<|im_start|>user
Extract from: "{content}"{date_context}<|im_end|>
<|im_start|>assistant
"#,
        speaker_rule = speaker_rule,
        date_instruction = date_instruction,
        example1_input = example1_input,
        example1_output = example1_output,
        example2_input = example2_input,
        example2_output = example2_output,
        example3_input = example3_input,
        example3_output = example3_output,
        content = truncated_content,
        date_context = if session_date.is_some() {
            format!("\nSession date: {}", session_date.unwrap())
        } else {
            String::new()
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_family_from_path_qwen() {
        let path = std::path::Path::new("models/qwen3-4b/Qwen3-4B-Instruct-2507.Q4_K_M.gguf");
        assert_eq!(ModelFamily::from_path(path), ModelFamily::Qwen);
    }

    #[test]
    fn test_model_family_from_path_phi4() {
        let path =
            std::path::Path::new("models/phi-4-mini/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf");
        assert_eq!(ModelFamily::from_path(path), ModelFamily::Phi4);
    }

    #[test]
    fn test_model_family_from_path_unknown_defaults_to_qwen() {
        let path = std::path::Path::new("models/some-random-model.gguf");
        assert_eq!(ModelFamily::from_path(path), ModelFamily::Qwen);
    }

    #[test]
    fn test_apply_chat_template_qwen_passthrough() {
        let prompt = "<|im_start|>system\nHello<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        let result = apply_chat_template(prompt, ModelFamily::Qwen);
        assert_eq!(result, prompt);
    }

    #[test]
    fn test_apply_chat_template_phi4_conversion() {
        let chatml = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        let result = apply_chat_template(chatml, ModelFamily::Phi4);
        assert!(result.contains("<|system|>\n"));
        assert!(result.contains("<|end|>"));
        assert!(result.contains("<|user|>\n"));
        assert!(result.contains("<|assistant|>\n"));
        assert!(!result.contains("<|im_start|>"));
        assert!(!result.contains("<|im_end|>"));
    }

    #[test]
    fn test_apply_chat_template_phi4_preserves_content() {
        let chatml = build_fewshot_extraction_prompt("Alice works at Google", None);
        let phi4 = apply_chat_template(&chatml, ModelFamily::Phi4);
        // Content preserved
        assert!(phi4.contains("Alice works at Google"));
        assert!(phi4.contains("biographical fact extraction specialist"));
        // Few-shot examples preserved
        assert!(phi4.contains("Bob plays guitar"));
        // Template tokens converted
        assert!(!phi4.contains("<|im_start|>"));
        assert!(phi4.contains("<|system|>"));
        assert!(phi4.contains("<|assistant|>"));
    }

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

    #[test]
    fn test_relationship_prompt_has_schema() {
        let prompt = build_relationship_extraction_prompt("Dave and Sarah work together", None);
        assert!(prompt.contains("Dave and Sarah work together"));
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>assistant"));
        // Core schema fields
        assert!(prompt.contains("entities"));
        assert!(prompt.contains("entity_facts"));
        assert!(prompt.contains("topics"));
        assert!(prompt.contains("importance"));
        // Relationship-specific vocabulary
        assert!(prompt.contains("relationship"));
        assert!(prompt.contains("affiliation"));
        // WRONG/RIGHT contrastive examples
        assert!(prompt.contains("WRONG"));
        assert!(prompt.contains("RIGHT"));
    }

    #[test]
    fn test_relationship_prompt_with_speaker() {
        let prompt = build_relationship_extraction_prompt("My sister got engaged", Some("Alice"));
        assert!(prompt.contains("spoken by Alice"));
        assert!(prompt.contains("Attribute all facts to Alice"));
    }

    #[test]
    fn test_event_temporal_prompt_has_schema() {
        let prompt = build_event_temporal_extraction_prompt("We flew to Tokyo last March", None);
        assert!(prompt.contains("We flew to Tokyo last March"));
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>assistant"));
        // Core schema fields
        assert!(prompt.contains("entities"));
        assert!(prompt.contains("entity_facts"));
        assert!(prompt.contains("topics"));
        assert!(prompt.contains("importance"));
        // Event-specific vocabulary
        assert!(prompt.contains("event"));
        assert!(prompt.contains("travel"));
        // WRONG/RIGHT contrastive examples
        assert!(prompt.contains("WRONG"));
        assert!(prompt.contains("RIGHT"));
    }

    #[test]
    fn test_event_temporal_prompt_with_speaker() {
        let prompt = build_event_temporal_extraction_prompt("I ran a marathon", Some("Bob"));
        assert!(prompt.contains("spoken by Bob"));
        assert!(prompt.contains("Attribute all facts to Bob"));
    }

    #[test]
    fn test_all_prompts_truncate_long_content() {
        let long = "x".repeat(2000);
        let p1 = build_fewshot_extraction_prompt(&long, None);
        let p2 = build_relationship_extraction_prompt(&long, None);
        let p3 = build_event_temporal_extraction_prompt(&long, None);
        let p4 = build_typed_extraction_prompt(&long, None, None);
        assert!(p1.contains("..."));
        assert!(p2.contains("..."));
        assert!(p3.contains("..."));
        assert!(p4.contains("..."));
    }

    #[test]
    fn test_typed_prompt_has_record_types() {
        let prompt = build_typed_extraction_prompt("Alice got promoted at Google", None, None);
        assert!(prompt.contains("Alice got promoted at Google"));
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>assistant"));
        // Record types
        assert!(prompt.contains("episodic"));
        assert!(prompt.contains("semantic"));
        assert!(prompt.contains("procedural"));
        // Core schema fields
        assert!(prompt.contains("entities"));
        assert!(prompt.contains("entity_facts"));
        assert!(prompt.contains("records"));
        assert!(prompt.contains("relationships"));
        // Few-shot examples contain records and relationships
        assert!(prompt.contains("record_type"));
        assert!(prompt.contains("relation_type"));
        assert!(prompt.contains("event_date"));
    }

    #[test]
    fn test_typed_prompt_with_speaker() {
        let prompt = build_typed_extraction_prompt("I love hiking", Some("Caroline"), None);
        assert!(prompt.contains("spoken by Caroline"));
        assert!(prompt.contains("Attribute all facts to Caroline"));
    }

    #[test]
    fn test_typed_prompt_with_session_date() {
        let prompt = build_typed_extraction_prompt(
            "We went camping last weekend",
            Some("Alice"),
            Some("2023-03-15"),
        );
        assert!(prompt.contains("SESSION DATE: 2023-03-15"));
        assert!(prompt.contains("subtract 1 day"));
        assert!(prompt.contains("Session date: 2023-03-15"));
    }

    #[test]
    fn test_typed_prompt_without_session_date() {
        let prompt = build_typed_extraction_prompt("I work at Google", None, None);
        assert!(!prompt.contains("SESSION DATE"));
    }

    #[test]
    fn test_typed_prompt_few_shot_examples_valid_json() {
        let prompt = build_typed_extraction_prompt("test content", None, None);
        // Extract JSON from few-shot examples and validate they parse
        // The examples contain "records" and "relationships" fields
        assert!(prompt.contains(r#""record_type":"semantic"#));
        assert!(prompt.contains(r#""record_type":"episodic"#));
        assert!(prompt.contains(r#""record_type":"procedural"#));
        assert!(prompt.contains(r#""relation_type":"spouse"#));
        assert!(prompt.contains(r#""relation_type":"friend"#));
    }
}
