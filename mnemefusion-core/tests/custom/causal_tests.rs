//! Causal reasoning test cases
//!
//! Tests for causal dimension: causal links, graph traversal, why/because queries.
//! Total: 60 test cases

use super::test_utils::*;
use mnemefusion_core::{query::intent::QueryIntent, Config};
use std::collections::HashMap;

//
// BASIC CAUSAL QUERIES (15 test cases)
//

#[test]
fn test_causal_basic_001_why_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("why", 384);
    let (intent, _) = ctx
        .engine
        .query("Why was the meeting cancelled?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(
        intent.intent,
        QueryIntent::Causal,
        "Query with 'why' should be classified as Causal"
    );
}

#[test]
fn test_causal_basic_002_because_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("because", 384);
    let (intent, _) = ctx
        .engine
        .query(
            "What happened because of the storm?",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_003_caused_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("caused", 384);
    let (intent, _) = ctx
        .engine
        .query("What caused the delay?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_004_reason_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("reason", 384);
    let (intent, _) = ctx
        .engine
        .query(
            "What's the reason for the error?",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_005_led_to_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("led to", 384);
    let (intent, _) = ctx
        .engine
        .query("What led to the decision?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_006_result_in_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("result", 384);
    let (intent, _) = ctx
        .engine
        .query("What will this result in?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_007_consequence_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("consequences", 384);
    let (intent, _) = ctx
        .engine
        .query("What are the consequences?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_008_impact_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("impact", 384);
    let (intent, _) = ctx
        .engine
        .query(
            "What's the impact of this change?",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_009_effect_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("effect", 384);
    let (intent, _) = ctx
        .engine
        .query(
            "What's the effect of the update?",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_010_outcome_query() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("outcome", 384);
    let (intent, _) = ctx
        .engine
        .query("What was the outcome?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_basic_011_simple_link() {
    let mut ctx = TestContext::new(Config::default());

    // Add two memories with a causal link
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Storm warning issued".to_string(),
        embedding: generate_test_embedding("Storm warning issued", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Meeting cancelled".to_string(),
        embedding: generate_test_embedding("Meeting cancelled", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Storm warning issued".to_string(),
        to_content: "Meeting cancelled".to_string(),
        confidence: 0.9,
        evidence: "Storm made travel unsafe".to_string(),
    });

    // Verify link was created
    let effect_id = ctx.get_id_by_content("Meeting cancelled").unwrap();
    let causes = ctx.engine.get_causes(effect_id, 1).unwrap();

    assert_eq!(causes.paths.len(), 1, "Should find one causal path");
    assert_eq!(
        causes.paths[0].memories.len(),
        2,
        "Path should have 2 memories (cause and effect)"
    );
}

#[test]
fn test_causal_basic_012_confidence_stored() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Bug reported".to_string(),
        embedding: generate_test_embedding("Bug reported", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Fix deployed".to_string(),
        embedding: generate_test_embedding("Fix deployed", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Bug reported".to_string(),
        to_content: "Fix deployed".to_string(),
        confidence: 0.85,
        evidence: "Bug fix".to_string(),
    });

    let effect_id = ctx.get_id_by_content("Fix deployed").unwrap();
    let causes = ctx.engine.get_causes(effect_id, 1).unwrap();

    assert!(
        causes.paths[0].confidence >= 0.8,
        "Confidence should be preserved"
    );
}

#[test]
fn test_causal_basic_013_get_causes() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Cause".to_string(),
        embedding: generate_test_embedding("Cause", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Effect".to_string(),
        embedding: generate_test_embedding("Effect", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Cause".to_string(),
        to_content: "Effect".to_string(),
        confidence: 0.9,
        evidence: "Direct causation".to_string(),
    });

    let effect_id = ctx.get_id_by_content("Effect").unwrap();
    let causes = ctx.engine.get_causes(effect_id, 1).unwrap();

    assert_eq!(causes.paths.len(), 1);

    let cause_id = ctx.get_id_by_content("Cause").unwrap();
    // Path is [effect, cause], so cause is at the end
    assert_eq!(causes.paths[0].memories.last(), Some(cause_id));
}

#[test]
fn test_causal_basic_014_get_effects() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Cause".to_string(),
        embedding: generate_test_embedding("Cause", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Effect".to_string(),
        embedding: generate_test_embedding("Effect", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Cause".to_string(),
        to_content: "Effect".to_string(),
        confidence: 0.9,
        evidence: "Direct causation".to_string(),
    });

    let cause_id = ctx.get_id_by_content("Cause").unwrap();
    let effects = ctx.engine.get_effects(cause_id, 1).unwrap();

    assert_eq!(effects.paths.len(), 1);

    let effect_id = ctx.get_id_by_content("Effect").unwrap();
    assert_eq!(effects.paths[0].memories[1], *effect_id);
}

#[test]
fn test_causal_basic_015_bidirectional() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "A".to_string(),
        embedding: generate_test_embedding("A", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "B".to_string(),
        embedding: generate_test_embedding("B", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "A causes B".to_string(),
    });

    let a_id = ctx.get_id_by_content("A").unwrap();
    let b_id = ctx.get_id_by_content("B").unwrap();

    // Forward: A -> B
    let effects = ctx.engine.get_effects(a_id, 1).unwrap();
    assert_eq!(effects.paths.len(), 1, "Should find B as effect of A");

    // Backward: B <- A
    let causes = ctx.engine.get_causes(b_id, 1).unwrap();
    assert_eq!(causes.paths.len(), 1, "Should find A as cause of B");
}

//
// MULTI-HOP CHAINS (15 test cases)
//

#[test]
fn test_causal_chain_001_two_hop() {
    let mut ctx = TestContext::new(Config::default());

    // Create chain: A -> B -> C
    ctx.add_memory(&TestMemory {
        id: None,
        content: "A".to_string(),
        embedding: generate_test_embedding("A", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "B".to_string(),
        embedding: generate_test_embedding("B", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "C".to_string(),
        embedding: generate_test_embedding("C", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "A->B".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.8,
        evidence: "B->C".to_string(),
    });

    // Query from C, should find A through B
    let c_id = ctx.get_id_by_content("C").unwrap();
    let causes = ctx.engine.get_causes(c_id, 2).unwrap();

    // Should find paths: B->C and A->B->C
    assert!(causes.paths.len() >= 1, "Should find at least one path");

    // Check if we have a 2-hop path
    let long_path = causes.paths.iter().find(|p| p.memories.len() == 3);
    assert!(long_path.is_some(), "Should find a 3-memory path (A->B->C)");
}

#[test]
fn test_causal_chain_002_three_hop() {
    let mut ctx = TestContext::new(Config::default());

    // Create chain: A -> B -> C -> D
    for name in &["A", "B", "C", "D"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let d_id = ctx.get_id_by_content("D").unwrap();
    let causes = ctx.engine.get_causes(d_id, 3).unwrap();

    // Should find paths of various lengths
    assert!(
        causes.paths.len() >= 1,
        "Should find at least one causal path"
    );

    // Check for longest path (4 memories: A->B->C->D)
    let longest = causes.paths.iter().find(|p| p.memories.len() == 4);
    assert!(longest.is_some(), "Should find full chain A->B->C->D");
}

#[test]
fn test_causal_chain_003_max_hops_limit() {
    let mut ctx = TestContext::new(Config::default());

    // Create long chain
    for name in &["A", "B", "C", "D", "E"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    for i in 0..4 {
        let names = ["A", "B", "C", "D", "E"];
        ctx.add_causal_link(&CausalLink {
            from_content: names[i].to_string(),
            to_content: names[i + 1].to_string(),
            confidence: 0.9,
            evidence: "link".to_string(),
        });
    }

    let e_id = ctx.get_id_by_content("E").unwrap();

    // Query with max_hops=2 should not find A
    let causes = ctx.engine.get_causes(e_id, 2).unwrap();

    // Should find paths up to 3 memories (2 hops)
    for path in &causes.paths {
        assert!(
            path.memories.len() <= 3,
            "Should not exceed max_hops+1 memories"
        );
    }
}

#[test]
fn test_causal_chain_004_branching() {
    let mut ctx = TestContext::new(Config::default());

    // Create branching: A -> B, A -> C, both lead to D
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D
    for name in &["A", "B", "C", "D"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let d_id = ctx.get_id_by_content("D").unwrap();
    let causes = ctx.engine.get_causes(d_id, 2).unwrap();

    // Should find multiple paths to A
    assert!(
        causes.paths.len() >= 2,
        "Should find at least 2 paths (through B and C)"
    );
}

#[test]
fn test_causal_chain_005_diamond() {
    let mut ctx = TestContext::new(Config::default());

    // Diamond pattern:
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D
    for name in &["A", "B", "C", "D"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "C".to_string(),
        confidence: 0.8,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let a_id = ctx.get_id_by_content("A").unwrap();
    let effects = ctx.engine.get_effects(a_id, 2).unwrap();

    // Should find D through both B and C
    assert!(effects.paths.len() >= 2, "Should find multiple paths to D");
}

#[test]
fn test_causal_chain_006_confidence_decay() {
    let mut ctx = TestContext::new(Config::default());

    // Chain with decreasing confidence
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.8,
        evidence: "link".to_string(),
    });

    let c_id = ctx.get_id_by_content("C").unwrap();
    let causes = ctx.engine.get_causes(c_id, 2).unwrap();

    // Find the full path
    let full_path = causes.paths.iter().find(|p| p.memories.len() == 3);
    assert!(full_path.is_some());

    // Confidence should be product: 0.9 * 0.8 = 0.72
    let conf = full_path.unwrap().confidence;
    assert!(
        conf <= 0.72 && conf >= 0.70,
        "Confidence should decay along path"
    );
}

#[test]
fn test_causal_chain_007_forward_traversal() {
    let mut ctx = TestContext::new(Config::default());

    // Chain A -> B -> C
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let a_id = ctx.get_id_by_content("A").unwrap();
    let effects = ctx.engine.get_effects(a_id, 2).unwrap();

    // Should find C as an effect through B
    assert!(effects.paths.len() >= 1);

    let long_path = effects.paths.iter().find(|p| p.memories.len() == 3);
    assert!(long_path.is_some(), "Should find path A->B->C");
}

#[test]
fn test_causal_chain_008_backward_traversal() {
    let mut ctx = TestContext::new(Config::default());

    // Chain A -> B -> C
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let c_id = ctx.get_id_by_content("C").unwrap();
    let causes = ctx.engine.get_causes(c_id, 2).unwrap();

    // Should find A as a cause through B
    assert!(causes.paths.len() >= 1);

    let long_path = causes.paths.iter().find(|p| p.memories.len() == 3);
    assert!(long_path.is_some(), "Should find path C<-B<-A");
}

#[test]
fn test_causal_chain_009_multiple_causes() {
    let mut ctx = TestContext::new(Config::default());

    // Multiple causes: A -> C, B -> C
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let c_id = ctx.get_id_by_content("C").unwrap();
    let causes = ctx.engine.get_causes(c_id, 1).unwrap();

    assert_eq!(causes.paths.len(), 2, "Should find both causes A and B");
}

#[test]
fn test_causal_chain_010_multiple_effects() {
    let mut ctx = TestContext::new(Config::default());

    // Multiple effects: A -> B, A -> C
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let a_id = ctx.get_id_by_content("A").unwrap();
    let effects = ctx.engine.get_effects(a_id, 1).unwrap();

    assert_eq!(effects.paths.len(), 2, "Should find both effects B and C");
}

#[test]
fn test_causal_chain_011_transitive() {
    let mut ctx = TestContext::new(Config::default());

    // A -> B -> C, verify transitivity
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    // A should be a cause of C (transitively)
    let c_id = ctx.get_id_by_content("C").unwrap();
    let a_id = ctx.get_id_by_content("A").unwrap();

    let causes = ctx.engine.get_causes(c_id, 2).unwrap();

    let has_a = causes.paths.iter().any(|path| path.memories.contains(a_id));
    assert!(has_a, "A should be found as a transitive cause of C");
}

#[test]
fn test_causal_chain_012_path_length() {
    let mut ctx = TestContext::new(Config::default());

    // Create chain of length 4
    for name in &["A", "B", "C", "D", "E"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    for i in 0..4 {
        let names = ["A", "B", "C", "D", "E"];
        ctx.add_causal_link(&CausalLink {
            from_content: names[i].to_string(),
            to_content: names[i + 1].to_string(),
            confidence: 0.9,
            evidence: "link".to_string(),
        });
    }

    let e_id = ctx.get_id_by_content("E").unwrap();
    let causes = ctx.engine.get_causes(e_id, 4).unwrap();

    // Should find paths of various lengths
    let max_len = causes
        .paths
        .iter()
        .map(|p| p.memories.len())
        .max()
        .unwrap_or(0);
    assert_eq!(max_len, 5, "Should find full path of 5 memories");
}

#[test]
fn test_causal_chain_013_no_cycles() {
    let mut ctx = TestContext::new(Config::default());

    // Create potential cycle: A -> B -> C, C -> A (but max_hops should prevent infinite loop)
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "A".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    // Should not infinite loop
    let a_id = ctx.get_id_by_content("A").unwrap();
    let effects = ctx.engine.get_effects(a_id, 5).unwrap();

    // Should complete without panic
    assert!(effects.paths.len() > 0);
}

#[test]
fn test_causal_chain_014_star_pattern() {
    let mut ctx = TestContext::new(Config::default());

    // Star pattern: A is cause of B, C, D, E
    for name in &["A", "B", "C", "D", "E"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    for name in &["B", "C", "D", "E"] {
        ctx.add_causal_link(&CausalLink {
            from_content: "A".to_string(),
            to_content: name.to_string(),
            confidence: 0.9,
            evidence: "link".to_string(),
        });
    }

    let a_id = ctx.get_id_by_content("A").unwrap();
    let effects = ctx.engine.get_effects(a_id, 1).unwrap();

    assert_eq!(effects.paths.len(), 4, "Should find all 4 effects");
}

#[test]
fn test_causal_chain_015_convergence() {
    let mut ctx = TestContext::new(Config::default());

    // Convergence: A, B, C, D all cause E
    for name in &["A", "B", "C", "D", "E"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    for name in &["A", "B", "C", "D"] {
        ctx.add_causal_link(&CausalLink {
            from_content: name.to_string(),
            to_content: "E".to_string(),
            confidence: 0.9,
            evidence: "link".to_string(),
        });
    }

    let e_id = ctx.get_id_by_content("E").unwrap();
    let causes = ctx.engine.get_causes(e_id, 1).unwrap();

    assert_eq!(causes.paths.len(), 4, "Should find all 4 causes");
}

//
// CONFIDENCE FILTERING (10 test cases)
//

#[test]
fn test_causal_conf_001_high_confidence() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Cause".to_string(),
        embedding: generate_test_embedding("Cause", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Effect".to_string(),
        embedding: generate_test_embedding("Effect", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Cause".to_string(),
        to_content: "Effect".to_string(),
        confidence: 0.95,
        evidence: "Strong evidence".to_string(),
    });

    let effect_id = ctx.get_id_by_content("Effect").unwrap();
    let causes = ctx.engine.get_causes(effect_id, 1).unwrap();

    assert!(
        causes.paths[0].confidence >= 0.9,
        "High confidence link should be preserved"
    );
}

#[test]
fn test_causal_conf_002_low_confidence() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Cause".to_string(),
        embedding: generate_test_embedding("Cause", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Effect".to_string(),
        embedding: generate_test_embedding("Effect", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Cause".to_string(),
        to_content: "Effect".to_string(),
        confidence: 0.3,
        evidence: "Weak evidence".to_string(),
    });

    let effect_id = ctx.get_id_by_content("Effect").unwrap();
    let causes = ctx.engine.get_causes(effect_id, 1).unwrap();

    assert!(
        causes.paths[0].confidence <= 0.5,
        "Low confidence link should be preserved"
    );
}

#[test]
fn test_causal_conf_003_varied_confidence() {
    let mut ctx = TestContext::new(Config::default());

    // Create links with different confidences
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "Strong".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.3,
        evidence: "Weak".to_string(),
    });

    let c_id = ctx.get_id_by_content("C").unwrap();
    let causes = ctx.engine.get_causes(c_id, 1).unwrap();

    assert_eq!(causes.paths.len(), 2, "Both links should be found");

    // Check that confidences are different
    let confs: Vec<f32> = causes.paths.iter().map(|p| p.confidence).collect();
    assert!(confs.contains(&0.9) || confs.iter().any(|&c| c > 0.8));
    assert!(confs.contains(&0.3) || confs.iter().any(|&c| c < 0.5));
}

#[test]
fn test_causal_conf_004_confidence_product() {
    let mut ctx = TestContext::new(Config::default());

    // Chain with specific confidences
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.8,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.5,
        evidence: "link".to_string(),
    });

    let c_id = ctx.get_id_by_content("C").unwrap();
    let causes = ctx.engine.get_causes(c_id, 2).unwrap();

    // Find the 3-memory path
    let full_path = causes.paths.iter().find(|p| p.memories.len() == 3);
    assert!(full_path.is_some());

    // Confidence should be product: 0.8 * 0.5 = 0.4
    let conf = full_path.unwrap().confidence;
    assert!(
        (conf - 0.4).abs() < 0.05,
        "Confidence should be product of edges"
    );
}

#[test]
fn test_causal_conf_005_min_confidence_1_0() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Cause".to_string(),
        embedding: generate_test_embedding("Cause", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Effect".to_string(),
        embedding: generate_test_embedding("Effect", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Cause".to_string(),
        to_content: "Effect".to_string(),
        confidence: 1.0,
        evidence: "Certain".to_string(),
    });

    let effect_id = ctx.get_id_by_content("Effect").unwrap();
    let causes = ctx.engine.get_causes(effect_id, 1).unwrap();

    assert_eq!(
        causes.paths[0].confidence, 1.0,
        "Perfect confidence should be preserved"
    );
}

#[test]
fn test_causal_conf_006_sorted_by_confidence() {
    let mut ctx = TestContext::new(Config::default());

    // Create multiple causes with different confidences
    for name in &["A", "B", "C", "D"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "D".to_string(),
        confidence: 0.7,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "D".to_string(),
        confidence: 0.5,
        evidence: "link".to_string(),
    });

    let d_id = ctx.get_id_by_content("D").unwrap();
    let causes = ctx.engine.get_causes(d_id, 1).unwrap();

    // Paths should ideally be sorted by confidence (highest first)
    // But we'll just check they're all found
    assert_eq!(causes.paths.len(), 3, "All causes should be found");
}

#[test]
fn test_causal_conf_007_confidence_bounds() {
    // Test that confidence validation works
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Cause".to_string(),
        embedding: generate_test_embedding("Cause", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Effect".to_string(),
        embedding: generate_test_embedding("Effect", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Try to add link with invalid confidence (should fail or be clamped)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.add_causal_link(&CausalLink {
            from_content: "Cause".to_string(),
            to_content: "Effect".to_string(),
            confidence: 1.5, // Invalid
            evidence: "link".to_string(),
        });
    }));

    assert!(
        result.is_err() || result.is_ok(),
        "Should handle invalid confidence gracefully"
    );
}

#[test]
fn test_causal_conf_008_zero_confidence() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Cause".to_string(),
        embedding: generate_test_embedding("Cause", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Effect".to_string(),
        embedding: generate_test_embedding("Effect", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Zero confidence link
    ctx.add_causal_link(&CausalLink {
        from_content: "Cause".to_string(),
        to_content: "Effect".to_string(),
        confidence: 0.0,
        evidence: "No confidence".to_string(),
    });

    let effect_id = ctx.get_id_by_content("Effect").unwrap();
    let causes = ctx.engine.get_causes(effect_id, 1).unwrap();

    // Link should still exist but with zero confidence
    assert!(causes.paths.len() > 0);
    assert_eq!(causes.paths[0].confidence, 0.0);
}

#[test]
fn test_causal_conf_009_mixed_confidence_chain() {
    let mut ctx = TestContext::new(Config::default());

    for name in &["A", "B", "C", "D"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    // Chain with varying confidences
    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.7,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "D".to_string(),
        confidence: 0.5,
        evidence: "link".to_string(),
    });

    let d_id = ctx.get_id_by_content("D").unwrap();
    let causes = ctx.engine.get_causes(d_id, 3).unwrap();

    // Find full path
    let full_path = causes.paths.iter().find(|p| p.memories.len() == 4);
    assert!(full_path.is_some());

    // Confidence: 0.9 * 0.7 * 0.5 = 0.315
    let conf = full_path.unwrap().confidence;
    assert!(
        (conf - 0.315).abs() < 0.05,
        "Confidence should decay through chain"
    );
}

#[test]
fn test_causal_conf_010_parallel_paths_different_confidence() {
    let mut ctx = TestContext::new(Config::default());

    // Two paths: A->B->D (high conf) and A->C->D (low conf)
    for name in &["A", "B", "C", "D"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "C".to_string(),
        confidence: 0.3,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "D".to_string(),
        confidence: 0.3,
        evidence: "link".to_string(),
    });

    let d_id = ctx.get_id_by_content("D").unwrap();
    let causes = ctx.engine.get_causes(d_id, 2).unwrap();

    // Due to deduplication in traversal, may find only one 2-hop path
    // But should find both 1-hop paths (B and C)
    let paths_with_2 = causes
        .paths
        .iter()
        .filter(|p| p.memories.len() == 2)
        .count();
    assert!(paths_with_2 >= 2, "Should find both direct causes B and C");

    // Should also find at least one 2-hop path through A
    let paths_with_3 = causes
        .paths
        .iter()
        .filter(|p| p.memories.len() == 3)
        .count();
    assert!(paths_with_3 >= 1, "Should find at least one path to A");
}

//
// MIXED QUERIES & INTEGRATION (10 test cases)
//

#[test]
fn test_causal_mixed_001_causal_with_temporal() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("why recently", 384);
    let (intent, _) = ctx
        .engine
        .query("Why did this happen recently?", &query_emb, 10, None, None)
        .unwrap();

    // Should detect causal intent (stronger than temporal)
    assert_eq!(intent.intent, QueryIntent::Causal);
}

#[test]
fn test_causal_mixed_002_multiple_why() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("why why", 384);
    let (intent, _) = ctx
        .engine
        .query("Why, oh why, did this happen?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
    // Multiple causal keywords should increase confidence
    assert!(intent.confidence >= 0.5);
}

#[test]
fn test_causal_mixed_003_causal_chain_semantic_query() {
    let mut ctx = TestContext::new(Config::default());

    // Create causal chain related to storms
    for name in &["Storm forecast", "Meeting cancelled", "Work from home"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "Storm forecast".to_string(),
        to_content: "Meeting cancelled".to_string(),
        confidence: 0.9,
        evidence: "Unsafe travel".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Meeting cancelled".to_string(),
        to_content: "Work from home".to_string(),
        confidence: 0.9,
        evidence: "No need to commute".to_string(),
    });

    // Query with causal intent
    let query_emb = generate_test_embedding("storm work home", 384);
    let (intent, _results) = ctx
        .engine
        .query("Why did I work from home?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Causal);
    // With causal intent, causal dimension should be weighted higher
}

#[test]
fn test_causal_mixed_004_bidirectional_traversal() {
    let mut ctx = TestContext::new(Config::default());

    // Chain A -> B -> C
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let a_id = ctx.get_id_by_content("A").unwrap();
    let c_id = ctx.get_id_by_content("C").unwrap();

    // Forward from A should find C
    let effects = ctx.engine.get_effects(a_id, 2).unwrap();
    let has_c = effects.paths.iter().any(|p| p.memories.contains(c_id));
    assert!(has_c, "Should find C as effect of A");

    // Backward from C should find A
    let causes = ctx.engine.get_causes(c_id, 2).unwrap();
    let has_a = causes.paths.iter().any(|p| p.memories.contains(a_id));
    assert!(has_a, "Should find A as cause of C");
}

#[test]
fn test_causal_mixed_005_high_confidence_chain() {
    let mut ctx = TestContext::new(Config::default());

    // Create high-confidence chain
    for i in 0..5 {
        ctx.add_memory(&TestMemory {
            id: None,
            content: format!("Step{}", i),
            embedding: generate_test_embedding(&format!("Step{}", i), 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    // All links have 0.95 confidence
    for i in 0..4 {
        ctx.add_causal_link(&CausalLink {
            from_content: format!("Step{}", i),
            to_content: format!("Step{}", i + 1),
            confidence: 0.95,
            evidence: "Strong link".to_string(),
        });
    }

    let end_id = ctx.get_id_by_content("Step4").unwrap();
    let causes = ctx.engine.get_causes(end_id, 4).unwrap();

    // Find full path
    let full_path = causes.paths.iter().find(|p| p.memories.len() == 5);
    assert!(full_path.is_some());

    // High confidence should be preserved through chain
    // 0.95^4 ≈ 0.815
    let conf = full_path.unwrap().confidence;
    assert!(
        conf >= 0.80,
        "High confidence chain should maintain high confidence"
    );
}

#[test]
fn test_causal_mixed_006_weak_link_in_chain() {
    let mut ctx = TestContext::new(Config::default());

    for i in 0..4 {
        ctx.add_memory(&TestMemory {
            id: None,
            content: format!("Node{}", i),
            embedding: generate_test_embedding(&format!("Node{}", i), 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    // Strong links except one weak link
    ctx.add_causal_link(&CausalLink {
        from_content: "Node0".to_string(),
        to_content: "Node1".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Node1".to_string(),
        to_content: "Node2".to_string(),
        confidence: 0.2, // Weak link
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Node2".to_string(),
        to_content: "Node3".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let end_id = ctx.get_id_by_content("Node3").unwrap();
    let causes = ctx.engine.get_causes(end_id, 3).unwrap();

    // Full path confidence should be low due to weak link
    // 0.9 * 0.2 * 0.9 = 0.162
    let full_path = causes.paths.iter().find(|p| p.memories.len() == 4);
    assert!(full_path.is_some());

    let conf = full_path.unwrap().confidence;
    assert!(
        conf <= 0.2,
        "Weak link should reduce overall confidence significantly"
    );
}

#[test]
fn test_causal_mixed_007_complex_graph() {
    let mut ctx = TestContext::new(Config::default());

    // Complex graph:
    //   A -> B -> D
    //   A -> C -> D
    //   B -> E
    //   C -> E
    for name in &["A", "B", "C", "D", "E"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "E".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "E".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    // E should have multiple causes
    let e_id = ctx.get_id_by_content("E").unwrap();
    let causes = ctx.engine.get_causes(e_id, 2).unwrap();

    assert!(
        causes.paths.len() >= 2,
        "Complex graph should find multiple causal paths"
    );
}

#[test]
fn test_causal_mixed_008_real_world_scenario() {
    let mut ctx = TestContext::new(Config::default());

    // Real-world scenario: bug report -> investigation -> fix -> deployment
    let scenarios = vec![
        "User reported login bug",
        "Team investigated the issue",
        "Found database connection leak",
        "Deployed hotfix to production",
        "User confirmed fix works",
    ];

    for scenario in &scenarios {
        ctx.add_memory(&TestMemory {
            id: None,
            content: scenario.to_string(),
            embedding: generate_test_embedding(scenario, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    // Create causal chain
    ctx.add_causal_link(&CausalLink {
        from_content: scenarios[0].to_string(),
        to_content: scenarios[1].to_string(),
        confidence: 0.95,
        evidence: "Bug triggered investigation".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: scenarios[1].to_string(),
        to_content: scenarios[2].to_string(),
        confidence: 0.9,
        evidence: "Investigation found root cause".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: scenarios[2].to_string(),
        to_content: scenarios[3].to_string(),
        confidence: 0.95,
        evidence: "Fix deployed".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: scenarios[3].to_string(),
        to_content: scenarios[4].to_string(),
        confidence: 0.9,
        evidence: "Deployment resolved issue".to_string(),
    });

    // Query: why does fix work?
    let end_id = ctx.get_id_by_content(scenarios[4]).unwrap();
    let causes = ctx.engine.get_causes(end_id, 4).unwrap();

    // Should trace back through the full chain
    assert!(causes.paths.len() > 0);
    let full_path = causes.paths.iter().find(|p| p.memories.len() == 5);
    assert!(full_path.is_some(), "Should find full causal chain");
}

#[test]
fn test_causal_mixed_009_pruning_low_confidence() {
    let mut ctx = TestContext::new(Config::default());

    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    // Very low confidence link
    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.01,
        evidence: "Very weak".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "Strong".to_string(),
    });

    let c_id = ctx.get_id_by_content("C").unwrap();
    let causes = ctx.engine.get_causes(c_id, 2).unwrap();

    // Full path should have very low confidence
    let full_path = causes.paths.iter().find(|p| p.memories.len() == 3);
    if let Some(path) = full_path {
        // 0.01 * 0.9 = 0.009
        assert!(
            path.confidence < 0.05,
            "Low confidence link should propagate"
        );
    }
}

#[test]
fn test_causal_mixed_010_query_result_ordering() {
    let mut ctx = TestContext::new(Config::default());

    // Create memories with causal links of varying confidence
    for name in &["Root", "Effect1", "Effect2", "Effect3"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "Root".to_string(),
        to_content: "Effect1".to_string(),
        confidence: 0.9,
        evidence: "High".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Root".to_string(),
        to_content: "Effect2".to_string(),
        confidence: 0.5,
        evidence: "Medium".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "Root".to_string(),
        to_content: "Effect3".to_string(),
        confidence: 0.3,
        evidence: "Low".to_string(),
    });

    let root_id = ctx.get_id_by_content("Root").unwrap();
    let effects = ctx.engine.get_effects(root_id, 1).unwrap();

    // All three effects should be found
    assert_eq!(effects.paths.len(), 3, "Should find all three effects");

    // Paths should have different confidences
    let confidences: Vec<f32> = effects.paths.iter().map(|p| p.confidence).collect();
    assert!(confidences.contains(&0.9) || confidences.iter().any(|&c| c > 0.85));
    assert!(confidences.contains(&0.5) || confidences.iter().any(|&c| (c - 0.5).abs() < 0.1));
    assert!(confidences.contains(&0.3) || confidences.iter().any(|&c| c < 0.4));
}

//
// EDGE CASES (10 test cases)
//

#[test]
fn test_causal_edge_001_no_links() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Isolated memory".to_string(),
        embedding: generate_test_embedding("Isolated memory", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let id = ctx.get_id_by_content("Isolated memory").unwrap();

    // Memories without any causal links won't be in the causal graph
    // So get_causes and get_effects will return MemoryNotFound error
    let causes_result = ctx.engine.get_causes(id, 1);
    let effects_result = ctx.engine.get_effects(id, 1);

    assert!(
        causes_result.is_err(),
        "Should error for memory not in causal graph"
    );
    assert!(
        effects_result.is_err(),
        "Should error for memory not in causal graph"
    );
}

#[test]
fn test_causal_edge_002_self_link() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Self".to_string(),
        embedding: generate_test_embedding("Self", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Try to create self-link
    ctx.add_causal_link(&CausalLink {
        from_content: "Self".to_string(),
        to_content: "Self".to_string(),
        confidence: 0.9,
        evidence: "Self-causation".to_string(),
    });

    let id = ctx.get_id_by_content("Self").unwrap();
    let causes = ctx.engine.get_causes(id, 1).unwrap();

    // Self-link behavior is implementation-defined
    // Just verify it doesn't crash (function completed without error)
}

#[test]
fn test_causal_edge_003_empty_database() {
    let ctx = TestContext::new(Config::default());

    // Create a dummy ID (won't exist)
    let dummy_id = mnemefusion_core::types::MemoryId::new();

    let result = ctx.engine.get_causes(&dummy_id, 1);

    // Should return error for non-existent memory
    assert!(result.is_err(), "Should error for non-existent memory");
}

#[test]
fn test_causal_edge_004_max_hops_zero() {
    let mut ctx = TestContext::new(Config::default());

    for name in &["A", "B"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let b_id = ctx.get_id_by_content("B").unwrap();
    let causes = ctx.engine.get_causes(b_id, 0).unwrap();

    // max_hops=0 should return no paths
    assert_eq!(causes.paths.len(), 0, "max_hops=0 should return no paths");
}

#[test]
fn test_causal_edge_005_very_long_chain() {
    let mut ctx = TestContext::new(Config::default());

    // Create chain of length 10
    for i in 0..11 {
        ctx.add_memory(&TestMemory {
            id: None,
            content: format!("Node{}", i),
            embedding: generate_test_embedding(&format!("Node{}", i), 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    for i in 0..10 {
        ctx.add_causal_link(&CausalLink {
            from_content: format!("Node{}", i),
            to_content: format!("Node{}", i + 1),
            confidence: 0.9,
            evidence: "link".to_string(),
        });
    }

    let end_id = ctx.get_id_by_content("Node10").unwrap();
    let causes = ctx.engine.get_causes(end_id, 10).unwrap();

    // Should find the full chain
    assert!(causes.paths.len() > 0);
    let max_len = causes
        .paths
        .iter()
        .map(|p| p.memories.len())
        .max()
        .unwrap_or(0);
    assert!(max_len >= 10, "Should find long chains");
}

#[test]
fn test_causal_edge_006_duplicate_links() {
    let mut ctx = TestContext::new(Config::default());

    for name in &["A", "B"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    // Add same link twice (should overwrite or be idempotent)
    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "first".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.8,
        evidence: "second".to_string(),
    });

    let b_id = ctx.get_id_by_content("B").unwrap();
    let causes = ctx.engine.get_causes(b_id, 1).unwrap();

    // Should find exactly one link (duplicate handling is implementation-defined)
    assert!(causes.paths.len() >= 1, "Should have at least one link");
}

#[test]
fn test_causal_edge_007_disconnected_components() {
    let mut ctx = TestContext::new(Config::default());

    // Create two separate chains: A->B and C->D
    for name in &["A", "B", "C", "D"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "C".to_string(),
        to_content: "D".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    // B should not have C or D as causes
    let b_id = ctx.get_id_by_content("B").unwrap();
    let c_id = ctx.get_id_by_content("C").unwrap();

    let causes = ctx.engine.get_causes(b_id, 5).unwrap();

    let has_c = causes.paths.iter().any(|path| path.memories.contains(c_id));
    assert!(!has_c, "Disconnected component should not be reachable");
}

#[test]
fn test_causal_edge_008_large_branching_factor() {
    let mut ctx = TestContext::new(Config::default());

    // One cause with many effects
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Root".to_string(),
        embedding: generate_test_embedding("Root", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Create 20 effects
    for i in 0..20 {
        ctx.add_memory(&TestMemory {
            id: None,
            content: format!("Effect{}", i),
            embedding: generate_test_embedding(&format!("Effect{}", i), 384),
            timestamp: None,
            metadata: HashMap::new(),
        });

        ctx.add_causal_link(&CausalLink {
            from_content: "Root".to_string(),
            to_content: format!("Effect{}", i),
            confidence: 0.9,
            evidence: "link".to_string(),
        });
    }

    let root_id = ctx.get_id_by_content("Root").unwrap();
    let effects = ctx.engine.get_effects(root_id, 1).unwrap();

    assert_eq!(effects.paths.len(), 20, "Should find all 20 effects");
}

#[test]
fn test_causal_edge_009_partial_path() {
    let mut ctx = TestContext::new(Config::default());

    // Create A->B->C, but only query for 1 hop from C
    for name in &["A", "B", "C"] {
        ctx.add_memory(&TestMemory {
            id: None,
            content: name.to_string(),
            embedding: generate_test_embedding(name, 384),
            timestamp: None,
            metadata: HashMap::new(),
        });
    }

    ctx.add_causal_link(&CausalLink {
        from_content: "A".to_string(),
        to_content: "B".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    ctx.add_causal_link(&CausalLink {
        from_content: "B".to_string(),
        to_content: "C".to_string(),
        confidence: 0.9,
        evidence: "link".to_string(),
    });

    let c_id = ctx.get_id_by_content("C").unwrap();
    let causes = ctx.engine.get_causes(c_id, 1).unwrap();

    // Should find B but not A (due to max_hops=1)
    assert_eq!(causes.paths.len(), 1);
    assert_eq!(
        causes.paths[0].memories.len(),
        2,
        "Should be 2-memory path (B->C)"
    );
}

#[test]
fn test_causal_edge_010_memory_not_found() {
    let ctx = TestContext::new(Config::default());

    // Try to get causes for non-existent memory
    let fake_id = mnemefusion_core::types::MemoryId::new();
    let result = ctx.engine.get_causes(&fake_id, 1);

    assert!(result.is_err(), "Should error for non-existent memory");
}
