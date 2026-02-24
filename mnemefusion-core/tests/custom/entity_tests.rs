//! Entity query test cases
//!
//! Tests for entity dimension: entity extraction, entity-centric queries, entity relationships.
//! Total: 35 test cases

use super::test_utils::*;
use mnemefusion_core::{query::intent::QueryIntent, types::Timestamp, Config};
use std::collections::HashMap;

// Helper to get current timestamp
fn now_timestamp() -> Timestamp {
    Timestamp::now()
}

// Helper to get timestamp N days ago
fn days_ago(days: u64) -> Timestamp {
    Timestamp::now().subtract_days(days)
}

// Helper to get timestamp N hours ago
fn hours_ago(hours: u64) -> Timestamp {
    let micros_per_hour = 60 * 60 * 1_000_000u64;
    let now_micros = Timestamp::now().as_micros();
    Timestamp::from_micros(now_micros.saturating_sub(hours * micros_per_hour))
}

// ============================================================================
// Basic Entity Queries (10 tests)
// Tests for intent detection with entity-related keywords
// ============================================================================

#[test]
fn test_entity_basic_001_about_query() {
    let mut ctx = TestContext::new(Config::default());

    // Add memories with entity
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice reviewed the code".to_string(),
        embedding: generate_test_embedding("Alice reviewed the code", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob approved the PR".to_string(),
        embedding: generate_test_embedding("Bob approved the PR", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("about Alice", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("Tell me about Alice", &query_embedding, 10, None, None)
        .unwrap();

    // Should detect entity intent
    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent for 'about Alice'"
    );
}

#[test]
fn test_entity_basic_002_regarding_query() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project Alpha is delayed".to_string(),
        embedding: generate_test_embedding("Project Alpha is delayed", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("regarding Project Alpha", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query(
            "Information regarding Project Alpha",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent for 'regarding Project'"
    );
}

#[test]
fn test_entity_basic_003_concerning_query() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Team Beta shipped features".to_string(),
        embedding: generate_test_embedding("Team Beta shipped features", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("concerning Team Beta", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query(
            "What concerning Team Beta?",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent for 'concerning Team'"
    );
}

#[test]
fn test_entity_basic_004_related_to_query() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Q1 budget was approved".to_string(),
        embedding: generate_test_embedding("Q1 budget was approved", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("related to Q1", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query(
            "Show me items related to Q1",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent for 'related to Q1'"
    );
}

#[test]
fn test_entity_basic_005_with_entity() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Charlie joined the meeting".to_string(),
        embedding: generate_test_embedding("Charlie joined the meeting", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("with Charlie", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("Meetings with Charlie", &query_embedding, 10, None, None)
        .unwrap();

    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent for 'with Charlie'"
    );
}

#[test]
fn test_entity_basic_006_involving_entity() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Diana filed the bug report".to_string(),
        embedding: generate_test_embedding("Diana filed the bug report", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("involving Diana", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("Issues involving Diana", &query_embedding, 10, None, None)
        .unwrap();

    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent for 'involving Diana'"
    );
}

#[test]
fn test_entity_basic_007_mention_query() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "AWS costs increased".to_string(),
        embedding: generate_test_embedding("AWS costs increased", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("mention AWS", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("Mention AWS", &query_embedding, 10, None, None)
        .unwrap();

    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent for 'mention AWS'"
    );
}

#[test]
fn test_entity_basic_008_capitalized_word_trigger() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "GitHub Actions workflow failed".to_string(),
        embedding: generate_test_embedding("GitHub Actions workflow failed", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Tell me about GitHub Actions", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query(
            "Tell me about GitHub Actions",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Capitalized words after 'about' should trigger entity intent
    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent with capitalized word"
    );
}

#[test]
fn test_entity_basic_009_multi_word_entity() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project Alpha Beta launched".to_string(),
        embedding: generate_test_embedding("Project Alpha Beta launched", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("about Project Alpha Beta", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query(
            "What about Project Alpha Beta?",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        intent.intent,
        QueryIntent::Entity,
        "Should detect entity intent for multi-word entity"
    );
}

#[test]
fn test_entity_basic_010_lowercase_not_entity() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Testing lowercase queries".to_string(),
        embedding: generate_test_embedding("Testing lowercase queries", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("about testing", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("about testing", &query_embedding, 10, None, None)
        .unwrap();

    // Lowercase word after 'about' should NOT trigger entity intent
    assert_ne!(
        intent.intent,
        QueryIntent::Entity,
        "Should not detect entity intent for lowercase word"
    );
}

// ============================================================================
// Entity Extraction (8 tests)
// Tests for automatic entity extraction from memory content
// ============================================================================

#[test]
fn test_entity_extract_001_single_name() {
    let mut ctx = TestContext::new(Config::default());

    // Add memory with single person name
    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice completed the task".to_string(),
        embedding: generate_test_embedding("Alice completed the task", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Get entities for this memory
    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Should extract "Alice"
    assert_eq!(entities.len(), 1, "Should extract one entity");
    assert_eq!(entities[0].name, "Alice");
    assert_eq!(entities[0].mention_count, 1);
}

#[test]
fn test_entity_extract_002_multiple_names() {
    let mut ctx = TestContext::new(Config::default());

    // Use periods to separate names so they're not treated as multi-word phrases
    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice worked here. Bob helped. Charlie joined.".to_string(),
        embedding: generate_test_embedding("Alice worked here. Bob helped. Charlie joined.", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Should extract all three names
    assert_eq!(entities.len(), 3, "Should extract three entities");
    let names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains(&"Alice".to_string()));
    assert!(names.contains(&"Bob".to_string()));
    assert!(names.contains(&"Charlie".to_string()));
}

#[test]
fn test_entity_extract_003_multi_word_entity() {
    let mut ctx = TestContext::new(Config::default());

    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "Project Alpha is on track".to_string(),
        embedding: generate_test_embedding("Project Alpha is on track", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Should extract "Project Alpha" as single entity
    assert_eq!(entities.len(), 1, "Should extract one multi-word entity");
    assert_eq!(entities[0].name, "Project Alpha");
}

#[test]
fn test_entity_extract_004_organizations() {
    let mut ctx = TestContext::new(Config::default());

    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "We work with Acme Corp and Beta Industries".to_string(),
        embedding: generate_test_embedding("We work with Acme Corp and Beta Industries", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Should extract both organizations
    assert_eq!(entities.len(), 2, "Should extract both organizations");
    let names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains(&"Acme Corp".to_string()));
    assert!(names.contains(&"Beta Industries".to_string()));
}

#[test]
fn test_entity_extract_005_filters_stop_words() {
    let mut ctx = TestContext::new(Config::default());

    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "The meeting was on Monday at Building C".to_string(),
        embedding: generate_test_embedding("The meeting was on Monday at Building C", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Should extract "Building C" but NOT "The" or "Monday"
    assert_eq!(entities.len(), 1, "Should extract only Building C");
    assert_eq!(entities[0].name, "Building C");
}

#[test]
fn test_entity_extract_006_acronyms() {
    let mut ctx = TestContext::new(Config::default());

    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "NASA and MIT are collaborating".to_string(),
        embedding: generate_test_embedding("NASA and MIT are collaborating", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Should extract acronyms
    assert_eq!(entities.len(), 2, "Should extract both acronyms");
    let names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains(&"NASA".to_string()));
    assert!(names.contains(&"MIT".to_string()));
}

#[test]
fn test_entity_extract_007_no_entities() {
    let mut ctx = TestContext::new(Config::default());

    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "nothing here is capitalized".to_string(),
        embedding: generate_test_embedding("nothing here is capitalized", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Should extract no entities
    assert_eq!(
        entities.len(),
        0,
        "Should extract no entities from lowercase content"
    );
}

#[test]
fn test_entity_extract_008_mention_count_increments() {
    let mut ctx = TestContext::new(Config::default());

    // Add two memories mentioning Alice
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice wrote the code".to_string(),
        embedding: generate_test_embedding("Alice wrote the code", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice reviewed the PR".to_string(),
        embedding: generate_test_embedding("Alice reviewed the PR", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Get all entities
    let all_entities = ctx.engine.list_entities().unwrap();

    // Find Alice
    let alice = all_entities.iter().find(|e| e.name == "Alice");
    assert!(alice.is_some(), "Alice should exist");
    assert_eq!(
        alice.unwrap().mention_count,
        2,
        "Alice should have 2 mentions"
    );
}

// ============================================================================
// Entity-Centric Queries (7 tests)
// Tests for retrieving memories by entity name
// ============================================================================

#[test]
fn test_entity_centric_001_get_entity_memories() {
    let mut ctx = TestContext::new(Config::default());

    // Add memories mentioning Alice
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice joined the team".to_string(),
        embedding: generate_test_embedding("Alice joined the team", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice completed the sprint".to_string(),
        embedding: generate_test_embedding("Alice completed the sprint", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob helped Alice with debugging".to_string(),
        embedding: generate_test_embedding("Bob helped Alice with debugging", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Get all memories about Alice
    let memories = ctx.engine.get_entity_memories("Alice").unwrap();

    assert_eq!(
        memories.len(),
        3,
        "Should find all memories mentioning Alice"
    );
    assert!(memories
        .iter()
        .any(|m| m.content.contains("joined the team")));
    assert!(memories
        .iter()
        .any(|m| m.content.contains("completed the sprint")));
    assert!(memories
        .iter()
        .any(|m| m.content.contains("helped Alice with debugging")));
}

#[test]
fn test_entity_centric_002_case_insensitive_lookup() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice sent an email".to_string(),
        embedding: generate_test_embedding("Alice sent an email", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Lookup should be case-insensitive
    let memories_lower = ctx.engine.get_entity_memories("alice").unwrap();
    let memories_upper = ctx.engine.get_entity_memories("ALICE").unwrap();
    let memories_mixed = ctx.engine.get_entity_memories("AlIcE").unwrap();

    assert_eq!(memories_lower.len(), 1, "Lowercase lookup should work");
    assert_eq!(memories_upper.len(), 1, "Uppercase lookup should work");
    assert_eq!(memories_mixed.len(), 1, "Mixed case lookup should work");
}

#[test]
fn test_entity_centric_003_multi_word_entity_lookup() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project Alpha launched successfully".to_string(),
        embedding: generate_test_embedding("Project Alpha launched successfully", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project Alpha needs more testing".to_string(),
        embedding: generate_test_embedding("Project Alpha needs more testing", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let memories = ctx.engine.get_entity_memories("Project Alpha").unwrap();

    assert_eq!(
        memories.len(),
        2,
        "Should find memories for multi-word entity"
    );
}

#[test]
fn test_entity_centric_004_entity_not_found() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice is here".to_string(),
        embedding: generate_test_embedding("Alice is here", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Query for non-existent entity
    let memories = ctx.engine.get_entity_memories("Bob").unwrap();

    assert_eq!(
        memories.len(),
        0,
        "Should return empty for non-existent entity"
    );
}

#[test]
fn test_entity_centric_005_multiple_entities_in_memory() {
    let mut ctx = TestContext::new(Config::default());

    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice and Bob worked on Project Gamma".to_string(),
        embedding: generate_test_embedding("Alice and Bob worked on Project Gamma", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Should extract Alice, Bob, and Project Gamma
    assert_eq!(entities.len(), 3, "Should extract all three entities");
    let names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains(&"Alice".to_string()));
    assert!(names.contains(&"Bob".to_string()));
    assert!(names.contains(&"Project Gamma".to_string()));
}

#[test]
fn test_entity_centric_006_organization_queries() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Acme Corp announced new product".to_string(),
        embedding: generate_test_embedding("Acme Corp announced new product", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Acme Corp hired 50 engineers".to_string(),
        embedding: generate_test_embedding("Acme Corp hired 50 engineers", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Beta Inc filed for IPO".to_string(),
        embedding: generate_test_embedding("Beta Inc filed for IPO", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let acme_memories = ctx.engine.get_entity_memories("Acme Corp").unwrap();
    let beta_memories = ctx.engine.get_entity_memories("Beta Inc").unwrap();

    assert_eq!(
        acme_memories.len(),
        2,
        "Should find both Acme Corp memories"
    );
    assert_eq!(beta_memories.len(), 1, "Should find Beta Inc memory");
}

#[test]
fn test_entity_centric_007_list_all_entities() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice and Bob met in Paris".to_string(),
        embedding: generate_test_embedding("Alice and Bob met in Paris", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Charlie works at Google".to_string(),
        embedding: generate_test_embedding("Charlie works at Google", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let all_entities = ctx.engine.list_entities().unwrap();

    // Should have: Alice, Bob, Paris, Charlie, Google
    assert_eq!(all_entities.len(), 5, "Should extract all unique entities");

    let names: Vec<String> = all_entities.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains(&"Alice".to_string()));
    assert!(names.contains(&"Bob".to_string()));
    assert!(names.contains(&"Paris".to_string()));
    assert!(names.contains(&"Charlie".to_string()));
    assert!(names.contains(&"Google".to_string()));
}

// ============================================================================
// Entity Relationships (5 tests)
// Tests for shared entities and co-occurrence patterns
// ============================================================================

#[test]
fn test_entity_rel_001_shared_entity() {
    let mut ctx = TestContext::new(Config::default());

    // Multiple memories mentioning same entity
    let id1 = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice wrote the spec".to_string(),
        embedding: generate_test_embedding("Alice wrote the spec", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    let id2 = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice implemented the feature".to_string(),
        embedding: generate_test_embedding("Alice implemented the feature", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    let id3 = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice fixed the bug".to_string(),
        embedding: generate_test_embedding("Alice fixed the bug", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let alice_memories = ctx.engine.get_entity_memories("Alice").unwrap();

    assert_eq!(
        alice_memories.len(),
        3,
        "Should find all memories with shared entity"
    );
    let memory_ids: Vec<_> = alice_memories.iter().map(|m| m.id.clone()).collect();
    assert!(memory_ids.contains(&id1));
    assert!(memory_ids.contains(&id2));
    assert!(memory_ids.contains(&id3));
}

#[test]
fn test_entity_rel_002_entity_co_occurrence() {
    let mut ctx = TestContext::new(Config::default());

    // Alice and Bob mentioned together (use periods to avoid multi-word extraction)
    let _id1 = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice and Bob pair programmed together".to_string(),
        embedding: generate_test_embedding("Alice and Bob pair programmed together", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    let _id2 = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice reviewed code from Bob".to_string(),
        embedding: generate_test_embedding("Alice reviewed code from Bob", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Alice mentioned alone
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice attended the conference".to_string(),
        embedding: generate_test_embedding("Alice attended the conference", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let alice_memories = ctx.engine.get_entity_memories("Alice").unwrap();
    let bob_memories = ctx.engine.get_entity_memories("Bob").unwrap();

    assert_eq!(alice_memories.len(), 3, "Alice appears in 3 memories");
    assert_eq!(bob_memories.len(), 2, "Bob appears in 2 memories");

    // Check co-occurrence: memories containing both Alice and Bob
    let co_occurring: Vec<_> = alice_memories
        .iter()
        .filter(|m| bob_memories.iter().any(|bm| bm.id == m.id))
        .collect();

    assert_eq!(
        co_occurring.len(),
        2,
        "Alice and Bob co-occur in 2 memories"
    );
}

#[test]
fn test_entity_rel_003_entity_in_multiple_contexts() {
    let mut ctx = TestContext::new(Config::default());

    // Same entity in different contexts (avoid sentence-start capitalization issues)
    ctx.add_memory(&TestMemory {
        id: None,
        content: "The city of Paris is beautiful".to_string(),
        embedding: generate_test_embedding("The city of Paris is beautiful", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "We visited Paris last month".to_string(),
        embedding: generate_test_embedding("We visited Paris last month", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "The Paris office opened recently".to_string(),
        embedding: generate_test_embedding("The Paris office opened recently", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let paris_memories = ctx.engine.get_entity_memories("Paris").unwrap();

    assert_eq!(
        paris_memories.len(),
        3,
        "Should find entity across different contexts"
    );
}

#[test]
fn test_entity_rel_004_entity_mention_count_tracking() {
    let mut ctx = TestContext::new(Config::default());

    // Add memories progressively
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project X started".to_string(),
        embedding: generate_test_embedding("Project X started", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.list_entities().unwrap();
    let project_x = entities.iter().find(|e| e.name == "Project X").unwrap();
    assert_eq!(project_x.mention_count, 1);

    // Add another mention
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project X is progressing well".to_string(),
        embedding: generate_test_embedding("Project X is progressing well", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.list_entities().unwrap();
    let project_x = entities.iter().find(|e| e.name == "Project X").unwrap();
    assert_eq!(project_x.mention_count, 2);
}

#[test]
fn test_entity_rel_005_many_to_many_relationships() {
    let mut ctx = TestContext::new(Config::default());

    // Create many-to-many: 3 people, 2 projects
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice works on Project Alpha".to_string(),
        embedding: generate_test_embedding("Alice works on Project Alpha", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob works on Project Alpha".to_string(),
        embedding: generate_test_embedding("Bob works on Project Alpha", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Charlie works on Project Beta".to_string(),
        embedding: generate_test_embedding("Charlie works on Project Beta", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice also works on Project Beta".to_string(),
        embedding: generate_test_embedding("Alice also works on Project Beta", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let alpha_memories = ctx.engine.get_entity_memories("Project Alpha").unwrap();
    let beta_memories = ctx.engine.get_entity_memories("Project Beta").unwrap();
    let alice_memories = ctx.engine.get_entity_memories("Alice").unwrap();

    assert_eq!(alpha_memories.len(), 2, "Project Alpha has 2 memories");
    assert_eq!(beta_memories.len(), 2, "Project Beta has 2 memories");
    assert_eq!(alice_memories.len(), 2, "Alice appears in 2 memories");
}

// ============================================================================
// Mixed Entity Queries (10 tests)
// Tests combining entity with temporal or causal dimensions
// ============================================================================

#[test]
fn test_entity_mixed_001_entity_plus_temporal() {
    let mut ctx = TestContext::new(Config::default());

    // Add memories at different times
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice sent email".to_string(),
        embedding: generate_test_embedding("Alice sent email", 384),
        timestamp: Some(days_ago(3)), // 3 days ago
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice made a commit".to_string(),
        embedding: generate_test_embedding("Alice made a commit", 384),
        timestamp: Some(days_ago(1)), // 1 day ago
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob sent email".to_string(),
        embedding: generate_test_embedding("Bob sent email", 384),
        timestamp: Some(days_ago(1)), // 1 day ago
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Alice last 2 days", 384);
    let (intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "What did Alice do in the last 2 days?",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Should detect mixed intent or prioritize one (or factual for "what" questions)
    // The query has both entity ("Alice") and temporal ("last 2 days")
    assert!(
        intent.intent == QueryIntent::Entity
            || intent.intent == QueryIntent::Temporal
            || intent.intent == QueryIntent::Factual,
        "Should detect entity, temporal, or factual intent for mixed query"
    );

    // Results should be filtered by relevance
    assert!(!results.is_empty(), "Should return relevant results");
}

#[test]
fn test_entity_mixed_002_entity_plus_causal() {
    let mut ctx = TestContext::new(Config::default());

    let alice_joined = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice joined the team".to_string(),
        embedding: generate_test_embedding("Alice joined the team", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    let code_improved = ctx.add_memory(&TestMemory {
        id: None,
        content: "Code quality improved".to_string(),
        embedding: generate_test_embedding("Code quality improved", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Link causal relationship
    ctx.add_causal_link(&CausalLink {
        from_content: "Alice joined the team".to_string(),
        to_content: "Code quality improved".to_string(),
        confidence: 0.8,
        evidence: "Alice is a senior engineer".to_string(),
    });

    let query_embedding = generate_test_embedding("because Alice joined", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query(
            "What happened because Alice joined?",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Query has both entity ("Alice") and causal ("because")
    // Should detect one or the other based on patterns
    assert!(
        intent.intent == QueryIntent::Causal || intent.intent == QueryIntent::Entity,
        "Should detect causal or entity intent"
    );
}

#[test]
fn test_entity_mixed_003_entity_in_recent_time() {
    let mut ctx = TestContext::new(Config::default());

    // Alice activities over time
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice fixed bug #123".to_string(),
        embedding: generate_test_embedding("Alice fixed bug #123", 384),
        timestamp: Some(days_ago(5)),
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice deployed to prod".to_string(),
        embedding: generate_test_embedding("Alice deployed to prod", 384),
        timestamp: Some(hours_ago(2)), // 2 hours ago
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob deployed to staging".to_string(),
        embedding: generate_test_embedding("Bob deployed to staging", 384),
        timestamp: Some(hours_ago(1)),
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Alice recently", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "What did Alice do recently?",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Should return results (fusion will rank appropriately)
    assert!(!results.is_empty(), "Should find recent Alice activities");
}

#[test]
fn test_entity_mixed_004_multiple_entities_query() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice and Bob collaborated".to_string(),
        embedding: generate_test_embedding("Alice and Bob collaborated", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice worked alone".to_string(),
        embedding: generate_test_embedding("Alice worked alone", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob worked alone".to_string(),
        embedding: generate_test_embedding("Bob worked alone", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Charlie joined later".to_string(),
        embedding: generate_test_embedding("Charlie joined later", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Alice and Bob interactions", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "Interactions between Alice and Bob",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Should return results mentioning both or either
    assert!(!results.is_empty(), "Should find interactions");
}

#[test]
fn test_entity_mixed_005_entity_with_factual() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice specializes in Rust programming".to_string(),
        embedding: generate_test_embedding("Alice specializes in Rust programming", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob specializes in Python".to_string(),
        embedding: generate_test_embedding("Bob specializes in Python", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Charlie knows Rust and Go".to_string(),
        embedding: generate_test_embedding("Charlie knows Rust and Go", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Who knows Rust", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("Who knows Rust?", &query_embedding, 10, None, None)
        .unwrap();

    // "Who" is an entity-related question word
    // Should likely detect as entity or factual
    assert!(
        intent.intent == QueryIntent::Entity || intent.intent == QueryIntent::Factual,
        "Should detect entity or factual intent for 'who' questions"
    );
}

#[test]
fn test_entity_mixed_006_entity_before_date() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice submitted proposal".to_string(),
        embedding: generate_test_embedding("Alice submitted proposal", 384),
        timestamp: Some(days_ago(10)), // 10 days ago
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice got approval".to_string(),
        embedding: generate_test_embedding("Alice got approval", 384),
        timestamp: Some(days_ago(2)), // 2 days ago
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Alice before last week", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "What did Alice do before last week?",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Should find the proposal from 10 days ago
    assert!(
        !results.is_empty(),
        "Should find Alice's activities before last week"
    );
}

#[test]
fn test_entity_mixed_007_entity_causal_chain() {
    let mut ctx = TestContext::new(Config::default());

    let alice_bug = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice found security bug".to_string(),
        embedding: generate_test_embedding("Alice found security bug", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    let patch_created = ctx.add_memory(&TestMemory {
        id: None,
        content: "Security patch created".to_string(),
        embedding: generate_test_embedding("Security patch created", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    let system_secure = ctx.add_memory(&TestMemory {
        id: None,
        content: "System is now secure".to_string(),
        embedding: generate_test_embedding("System is now secure", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Create causal chain
    ctx.add_causal_link(&CausalLink {
        from_content: "Alice found security bug".to_string(),
        to_content: "Security patch created".to_string(),
        confidence: 0.9,
        evidence: "Direct response to bug".to_string(),
    });
    ctx.add_causal_link(&CausalLink {
        from_content: "Security patch created".to_string(),
        to_content: "System is now secure".to_string(),
        confidence: 0.85,
        evidence: "Patch fixed vulnerability".to_string(),
    });

    let query_embedding = generate_test_embedding("Alice discovery results", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "What resulted from Alice's discovery?",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Should return relevant results (fusion handles multi-dimensional ranking)
    assert!(
        !results.is_empty(),
        "Should find causal chain from Alice's action"
    );
}

#[test]
fn test_entity_mixed_008_entity_with_location() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice visited San Francisco office".to_string(),
        embedding: generate_test_embedding("Alice visited San Francisco office", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob visited San Francisco office".to_string(),
        embedding: generate_test_embedding("Bob visited San Francisco office", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice visited New York office".to_string(),
        embedding: generate_test_embedding("Alice visited New York office", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Alice San Francisco", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query("Alice in San Francisco", &query_embedding, 10, None, None)
        .unwrap();

    // Should find memories with both Alice and San Francisco
    assert!(
        !results.is_empty(),
        "Should find entity-location combinations"
    );
}

#[test]
fn test_entity_mixed_009_project_timeline() {
    let mut ctx = TestContext::new(Config::default());

    // Project timeline
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project Omega kicked off".to_string(),
        embedding: generate_test_embedding("Project Omega kicked off", 384),
        timestamp: Some(days_ago(30)),
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project Omega hit milestone 1".to_string(),
        embedding: generate_test_embedding("Project Omega hit milestone 1", 384),
        timestamp: Some(days_ago(15)),
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project Omega completed".to_string(),
        embedding: generate_test_embedding("Project Omega completed", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Project Omega progress timeline", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "Project Omega progress over time",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Should find all project memories
    assert_eq!(results.len(), 3, "Should find all project timeline entries");
}

#[test]
fn test_entity_mixed_010_real_world_scenario() {
    let mut ctx = TestContext::new(Config::default());

    // Real scenario: customer interaction tracking
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Acme Corp submitted support ticket".to_string(),
        embedding: generate_test_embedding("Acme Corp submitted support ticket", 384),
        timestamp: Some(days_ago(5)),
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice assigned to Acme Corp ticket".to_string(),
        embedding: generate_test_embedding("Alice assigned to Acme Corp ticket", 384),
        timestamp: Some(days_ago(4)),
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice diagnosed Acme Corp issue".to_string(),
        embedding: generate_test_embedding("Alice diagnosed Acme Corp issue", 384),
        timestamp: Some(days_ago(3)),
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Fix deployed for Acme Corp".to_string(),
        embedding: generate_test_embedding("Fix deployed for Acme Corp", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Acme Corp confirmed issue resolved".to_string(),
        embedding: generate_test_embedding("Acme Corp confirmed issue resolved", 384),
        timestamp: Some(hours_ago(2)),
        metadata: HashMap::new(),
    });

    // Query for company
    let acme_memories = ctx.engine.get_entity_memories("Acme Corp").unwrap();
    assert_eq!(
        acme_memories.len(),
        5,
        "Should track all customer interactions"
    );

    // Query for engineer
    let alice_memories = ctx.engine.get_entity_memories("Alice").unwrap();
    assert_eq!(
        alice_memories.len(),
        2,
        "Should track engineer's involvement"
    );

    // Mixed query
    let query_embedding = generate_test_embedding("Alice Acme Corp work", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "Alice's work on Acme Corp issue",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    assert!(
        !results.is_empty(),
        "Should find relevant customer-engineer interactions"
    );
}

// ============================================================================
// Edge Cases (5 tests)
// Tests for edge cases and error conditions
// ============================================================================

#[test]
fn test_entity_edge_001_empty_entity_name() {
    let ctx = TestContext::new(Config::default());

    // Query with empty entity name
    let memories = ctx.engine.get_entity_memories("").unwrap();

    assert_eq!(
        memories.len(),
        0,
        "Empty entity name should return no results"
    );
}

#[test]
fn test_entity_edge_002_special_characters() {
    let mut ctx = TestContext::new(Config::default());

    // Entities with special characters (extractor should clean them)
    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "Dr. Smith and Ms. Johnson collaborated".to_string(),
        embedding: generate_test_embedding("Dr. Smith and Ms. Johnson collaborated", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let entities = ctx.engine.get_memory_entities(&id).unwrap();

    // Punctuation should be stripped, multi-word phrases extracted
    // Expected: "Dr", "Smith", "Ms", "Johnson" (as separate words or phrases depending on extraction logic)
    // Or "Dr. Smith" and "Ms. Johnson" if extractor handles titles
    // At minimum should have Smith and Johnson
    assert!(!entities.is_empty(), "Should extract some entities");
    let names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    assert!(
        names.iter().any(|n| n.contains("Smith")),
        "Should extract Smith"
    );
    assert!(
        names.iter().any(|n| n.contains("Johnson")),
        "Should extract Johnson"
    );
}

#[test]
fn test_entity_edge_003_orphaned_entity_cleanup() {
    let mut ctx = TestContext::new(Config::default());

    // Add memory with entity
    let id = ctx.add_memory(&TestMemory {
        id: None,
        content: "TemporaryEntity exists".to_string(),
        embedding: generate_test_embedding("TemporaryEntity exists", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Verify entity exists
    let entities_before = ctx.engine.list_entities().unwrap();
    assert!(entities_before.iter().any(|e| e.name == "TemporaryEntity"));

    // Delete memory
    ctx.engine.delete(&id, None).unwrap();

    // Verify entity is cleaned up (orphaned)
    let entities_after = ctx.engine.list_entities().unwrap();
    assert!(
        !entities_after.iter().any(|e| e.name == "TemporaryEntity"),
        "Orphaned entity should be removed"
    );
}

#[test]
fn test_entity_edge_004_entity_deduplication() {
    let mut ctx = TestContext::new(Config::default());

    // Add multiple memories mentioning Alice (use mid-sentence or lowercase start)
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice completed the first task".to_string(),
        embedding: generate_test_embedding("Alice completed the first task", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "later, Alice completed another task".to_string(),
        embedding: generate_test_embedding("later, Alice completed another task", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "yesterday Alice completed final task".to_string(),
        embedding: generate_test_embedding("yesterday Alice completed final task", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let all_entities = ctx.engine.list_entities().unwrap();

    // Should have only one Alice entity (deduplication)
    let alice_entities: Vec<_> = all_entities
        .iter()
        .filter(|e| e.name.to_lowercase() == "alice")
        .collect();

    assert_eq!(
        alice_entities.len(),
        1,
        "Should deduplicate Alice across mentions"
    );
    assert_eq!(
        alice_entities[0].mention_count, 3,
        "Should track all 3 mentions"
    );
}

#[test]
fn test_entity_edge_005_no_extraction_when_disabled() {
    // Create engine with entity extraction disabled
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("test.mfdb");

    let mut config = Config::default();
    config.entity_extraction_enabled = false;

    let engine = mnemefusion_core::MemoryEngine::open(&path, config).unwrap();

    // Add memory with potential entities
    let embedding = generate_test_embedding("Alice and Bob worked together", 384);
    let id = engine
        .add(
            "Alice and Bob worked together".to_string(),
            embedding,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Get entities for this memory
    let entities = engine.get_memory_entities(&id).unwrap();

    assert_eq!(
        entities.len(),
        0,
        "Should not extract entities when disabled"
    );

    // List all entities
    let all_entities = engine.list_entities().unwrap();
    assert_eq!(
        all_entities.len(),
        0,
        "No entities should exist when extraction disabled"
    );
}

// ============================================================================
// Entity Query Results (2 tests)
// Tests for entity dimension in query results
// ============================================================================

#[test]
fn test_entity_query_001_entity_intent_returns_relevant() {
    let mut ctx = TestContext::new(Config::default());

    // Add diverse memories
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice completed project X".to_string(),
        embedding: generate_test_embedding("Alice completed project X", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob completed project Y".to_string(),
        embedding: generate_test_embedding("Bob completed project Y", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "General project information".to_string(),
        embedding: generate_test_embedding("General project information", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Tell me about Alice", 384);
    let (intent, results, _profile_ctx) = ctx
        .engine
        .query("Tell me about Alice", &query_embedding, 10, None, None)
        .unwrap();

    // Should detect entity intent
    assert_eq!(intent.intent, QueryIntent::Entity);

    // Should return relevant results (fusion will prioritize Alice-related)
    assert!(
        !results.is_empty(),
        "Should return results for entity query"
    );
}

#[test]
fn test_entity_query_002_mixed_entity_temporal() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice worked on feature".to_string(),
        embedding: generate_test_embedding("Alice worked on feature", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice attended meeting".to_string(),
        embedding: generate_test_embedding("Alice attended meeting", 384),
        timestamp: Some(days_ago(10)),
        metadata: HashMap::new(),
    });

    let query_embedding = generate_test_embedding("Alice yesterday", 384);
    let (intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "What did Alice do yesterday?",
            &query_embedding,
            10,
            None,
            None,
        )
        .unwrap();

    // Mixed query should balance dimensions
    // Could be Entity or Temporal depending on classifier priority
    assert!(
        intent.intent == QueryIntent::Entity || intent.intent == QueryIntent::Temporal,
        "Mixed query should detect either entity or temporal intent"
    );

    // Should still return results
    assert!(!results.is_empty(), "Should return results for mixed query");
}
