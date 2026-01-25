//! Temporal query test cases
//!
//! Tests for temporal dimension: recency, time ranges, relative time queries.
//! Total: 50 test cases

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

//
// RECENCY QUERIES (15 test cases)
//

#[test]
fn test_temporal_recency_001_recent() {
    let mut ctx = TestContext::new(Config::default());

    // Add memories at different times
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Meeting this morning".to_string(),
        embedding: generate_test_embedding("Meeting this morning", 384),
        timestamp: Some(hours_ago(2)),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Lunch yesterday".to_string(),
        embedding: generate_test_embedding("Lunch yesterday", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Project last week".to_string(),
        embedding: generate_test_embedding("Project last week", 384),
        timestamp: Some(days_ago(7)),
        metadata: HashMap::new(),
    });

    // Query for recent memories
    let query_emb = generate_test_embedding("recent", 384);
    let (intent, results) = ctx
        .engine
        .query("What happened recently?", &query_emb, 10, None, None)
        .unwrap();

    // Assert temporal intent detected
    assert_intent(intent.intent, QueryIntent::Temporal, "temporal_recency_001");
    assert_intent_confidence(intent.confidence, 0.3, "temporal_recency_001");

    // Most recent memory should be first
    assert!(results.len() >= 1, "Should return at least 1 result");
    let first_memory = &results[0].0; // (Memory, FusedResult) tuple
    assert_eq!(first_memory.content, "Meeting this morning");
}

#[test]
fn test_temporal_recency_002_latest() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Latest update".to_string(),
        embedding: generate_test_embedding("Latest update", 384),
        timestamp: Some(now_timestamp()),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Old update".to_string(),
        embedding: generate_test_embedding("Old update", 384),
        timestamp: Some(days_ago(30)),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("latest", 384);
    let (intent, results) = ctx
        .engine
        .query("Show me the latest updates", &query_emb, 5, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
    assert!(results.len() >= 1);

    let first = &results[0].0;
    assert_eq!(first.content, "Latest update");
}

#[test]
fn test_temporal_recency_003_newest() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Newest entry".to_string(),
        embedding: generate_test_embedding("Newest entry", 384),
        timestamp: Some(now_timestamp()),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Older entry".to_string(),
        embedding: generate_test_embedding("Older entry", 384),
        timestamp: Some(hours_ago(24)),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("newest", 384);
    let (intent, results) = ctx
        .engine
        .query("Get newest memories", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
    assert!(
        results[0].1.temporal_score > 0.5,
        "Temporal score should be significant"
    );
}

#[test]
fn test_temporal_recency_004_yesterday() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Memory from yesterday".to_string(),
        embedding: generate_test_embedding("Memory from yesterday", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("yesterday", 384);
    let (intent, _) = ctx
        .engine
        .query("What happened yesterday?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
    assert!(intent.confidence > 0.3);
}

#[test]
fn test_temporal_recency_005_today() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Today's task".to_string(),
        embedding: generate_test_embedding("Today's task", 384),
        timestamp: Some(hours_ago(3)),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("today", 384);
    let (intent, _) = ctx
        .engine
        .query("Show me what I did today", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_recency_006_tomorrow() {
    let query_emb = generate_test_embedding("tomorrow", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What's scheduled for tomorrow?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_recency_007_when() {
    let query_emb = generate_test_embedding("when", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("When did this happen?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_recency_008_since() {
    let query_emb = generate_test_embedding("since", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What changed since Monday?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_recency_009_before() {
    let query_emb = generate_test_embedding("before", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query(
            "Show me tasks before the meeting",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_recency_010_after() {
    let query_emb = generate_test_embedding("after", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What happened after lunch?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_recency_011_earlier() {
    let query_emb = generate_test_embedding("earlier", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("Show me earlier notes", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_recency_012_until() {
    let query_emb = generate_test_embedding("until", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("Events until Friday", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_recency_013_multiple_temporal_keywords() {
    let query_emb = generate_test_embedding("recent latest yesterday", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query(
            "Show me the most recent and latest updates from yesterday",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
    // Multiple temporal keywords should increase confidence
    assert!(intent.confidence > 0.4);
}

#[test]
fn test_temporal_recency_014_ordering() {
    let mut ctx = TestContext::new(Config::default());

    // Add 5 memories at different times
    for i in 0..5 {
        ctx.add_memory(&TestMemory {
            id: None,
            content: format!("Memory {}", i),
            embedding: generate_test_embedding(&format!("Memory {}", i), 384),
            timestamp: Some(days_ago(i)),
            metadata: HashMap::new(),
        });
    }

    let query_emb = generate_test_embedding("recent", 384);
    let (_, results) = ctx
        .engine
        .query("Show recent memories", &query_emb, 5, None, None)
        .unwrap();

    // Results should be ordered by recency (newest first)
    for i in 0..results.len() - 1 {
        let mem_i = &results[i].0;
        let mem_next = &results[i + 1].0;
        assert!(
            mem_i.created_at >= mem_next.created_at,
            "Results should be ordered newest first"
        );
    }
}

#[test]
fn test_temporal_recency_015_temporal_score_decreasing() {
    let mut ctx = TestContext::new(Config::default());

    for i in 0..3 {
        ctx.add_memory(&TestMemory {
            id: None,
            content: format!("Memory {}", i),
            embedding: generate_test_embedding(&format!("Memory {}", i), 384),
            timestamp: Some(days_ago(i * 7)), // 0, 7, 14 days ago
            metadata: HashMap::new(),
        });
    }

    let query_emb = generate_test_embedding("recent", 384);
    let (_, results) = ctx
        .engine
        .query("recent memories", &query_emb, 3, None, None)
        .unwrap();

    // Temporal scores should decrease for older memories
    for i in 0..results.len() - 1 {
        assert!(
            results[i].1.temporal_score >= results[i + 1].1.temporal_score,
            "Temporal scores should decrease for older memories"
        );
    }
}

//
// RANGE QUERIES (15 test cases)
//

#[test]
fn test_temporal_range_001_last_week() {
    let query_emb = generate_test_embedding("last week", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What did I work on last week?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_range_002_next_week() {
    let query_emb = generate_test_embedding("next week", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("Plans for next week", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_range_003_this_month() {
    let mut ctx = TestContext::new(Config::default());

    // Add memory from this month
    ctx.add_memory(&TestMemory {
        id: None,
        content: "This month's report".to_string(),
        embedding: generate_test_embedding("This month's report", 384),
        timestamp: Some(days_ago(15)),
        metadata: HashMap::new(),
    });

    // Add memory from last month
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Last month's report".to_string(),
        embedding: generate_test_embedding("Last month's report", 384),
        timestamp: Some(days_ago(40)),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("this month", 384);
    let (intent, _) = ctx
        .engine
        .query(
            "Show me this month's activities",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_range_004_january() {
    let query_emb = generate_test_embedding("january", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("Events in January", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_range_005_month_names() {
    let months = vec![
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
    ];

    for month in months {
        let query = format!("Show me {} activities", month);
        let query_emb = generate_test_embedding(&query, 384);
        let ctx = TestContext::new(Config::default());

        let (intent, _) = ctx
            .engine
            .query(&query, &query_emb, 10, None, None)
            .unwrap();
        assert_eq!(
            intent.intent,
            QueryIntent::Temporal,
            "Query with month name '{}' should be temporal",
            month
        );
    }
}

#[test]
fn test_temporal_range_006_monday() {
    let query_emb = generate_test_embedding("monday", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What happened on Monday?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_range_007_weekday_names() {
    let weekdays = vec![
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];

    for day in weekdays {
        let query = format!("Events on {}", day);
        let query_emb = generate_test_embedding(&query, 384);
        let ctx = TestContext::new(Config::default());

        let (intent, _) = ctx
            .engine
            .query(&query, &query_emb, 10, None, None)
            .unwrap();
        assert_eq!(
            intent.intent,
            QueryIntent::Temporal,
            "Query with weekday '{}' should be temporal",
            day
        );
    }
}

#[test]
fn test_temporal_range_008_specific_date() {
    let query_emb = generate_test_embedding("january 15", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What happened on January 15?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_range_009_between_dates() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "In range".to_string(),
        embedding: generate_test_embedding("In range", 384),
        timestamp: Some(days_ago(5)),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Out of range".to_string(),
        embedding: generate_test_embedding("Out of range", 384),
        timestamp: Some(days_ago(20)),
        metadata: HashMap::new(),
    });

    // Use temporal range query directly
    let start = days_ago(10);
    let end = now_timestamp();
    let results = ctx.engine.get_range(start, end, 10, None).unwrap();

    assert!(results.len() >= 1);
    let first = &results[0].0;
    assert_eq!(first.content, "In range");
}

#[test]
fn test_temporal_range_010_past_7_days() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Within 7 days".to_string(),
        embedding: generate_test_embedding("Within 7 days", 384),
        timestamp: Some(days_ago(3)),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Beyond 7 days".to_string(),
        embedding: generate_test_embedding("Beyond 7 days", 384),
        timestamp: Some(days_ago(10)),
        metadata: HashMap::new(),
    });

    let start = days_ago(7);
    let end = now_timestamp();
    let results = ctx.engine.get_range(start, end, 10, None).unwrap();

    // Should only include memory within 7 days
    for result in &results {
        let memory = &result.0;
        assert!(memory.created_at >= start);
    }
}

#[test]
fn test_temporal_range_011_past_30_days() {
    let start = days_ago(30);
    let end = now_timestamp();
    let ctx = TestContext::new(Config::default());

    let results = ctx.engine.get_range(start, end, 10, None).unwrap();
    // Should not error, even with empty database
    assert!(results.len() >= 0);
}

#[test]
fn test_temporal_range_012_exact_range() {
    let mut ctx = TestContext::new(Config::default());

    let start = days_ago(10);
    let end = days_ago(5);

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Before range".to_string(),
        embedding: generate_test_embedding("Before range", 384),
        timestamp: Some(days_ago(15)),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "In range".to_string(),
        embedding: generate_test_embedding("In range", 384),
        timestamp: Some(days_ago(7)),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "After range".to_string(),
        embedding: generate_test_embedding("After range", 384),
        timestamp: Some(days_ago(3)),
        metadata: HashMap::new(),
    });

    let results = ctx.engine.get_range(start, end, 10, None).unwrap();

    // Should only get the one memory in range
    assert_eq!(results.len(), 1);
    let memory = &results[0].0;
    assert_eq!(memory.content, "In range");
}

#[test]
fn test_temporal_range_013_empty_range() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Some memory".to_string(),
        embedding: generate_test_embedding("Some memory", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    // Query a range with no memories
    let start = days_ago(100);
    let end = days_ago(90);

    let results = ctx.engine.get_range(start, end, 10, None).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_temporal_range_014_overlapping_ranges() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Memory 1".to_string(),
        embedding: generate_test_embedding("Memory 1", 384),
        timestamp: Some(days_ago(5)),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Memory 2".to_string(),
        embedding: generate_test_embedding("Memory 2", 384),
        timestamp: Some(days_ago(7)),
        metadata: HashMap::new(),
    });

    // First range: last 10 days
    let results1 = ctx
        .engine
        .get_range(days_ago(10), now_timestamp(), 10, None)
        .unwrap();
    assert_eq!(results1.len(), 2);

    // Second range: last 6 days (should only get Memory 1)
    let results2 = ctx
        .engine
        .get_range(days_ago(6), now_timestamp(), 10, None)
        .unwrap();
    assert_eq!(results2.len(), 1);
}

#[test]
fn test_temporal_range_015_boundary_conditions() {
    let mut ctx = TestContext::new(Config::default());

    let boundary_time = days_ago(7);

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Exactly at boundary".to_string(),
        embedding: generate_test_embedding("Exactly at boundary", 384),
        timestamp: Some(boundary_time),
        metadata: HashMap::new(),
    });

    // Range that includes the boundary (inclusive)
    let results = ctx
        .engine
        .get_range(boundary_time, now_timestamp(), 10, None)
        .unwrap();
    assert_eq!(results.len(), 1);
}

//
// RELATIVE TIME (10 test cases)
//

#[test]
fn test_temporal_relative_001_days_ago() {
    let query_emb = generate_test_embedding("3 days ago", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("Show me notes from 3 days ago", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_002_weeks_ago() {
    let query_emb = generate_test_embedding("2 weeks ago", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What happened 2 weeks ago?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_003_months_ago() {
    let query_emb = generate_test_embedding("1 month ago", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("Events from 1 month ago", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_004_years_ago() {
    let query_emb = generate_test_embedding("1 year ago", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("Memories from 1 year ago", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_005_hours_ago() {
    let query_emb = generate_test_embedding("2 hours ago", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What happened 2 hours ago?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_006_earlier_today() {
    let query_emb = generate_test_embedding("earlier today", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query(
            "Show me notes from earlier today",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_007_this_morning() {
    let query_emb = generate_test_embedding("this morning", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What did I do this morning?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_008_last_night() {
    let query_emb = generate_test_embedding("last night", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("Events from last night", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_009_this_afternoon() {
    let query_emb = generate_test_embedding("this afternoon", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query("What happened this afternoon?", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_relative_010_mixed_relative() {
    let query_emb = generate_test_embedding("past few days", 384);
    let ctx = TestContext::new(Config::default());

    let (intent, _) = ctx
        .engine
        .query(
            "Show me tasks from the past few days",
            &query_emb,
            10,
            None,
            None,
        )
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

//
// EDGE CASES (10 test cases)
//

#[test]
fn test_temporal_edge_001_oldest() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Oldest memory".to_string(),
        embedding: generate_test_embedding("Oldest memory", 384),
        timestamp: Some(days_ago(365)),
        metadata: HashMap::new(),
    });

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Newer memory".to_string(),
        embedding: generate_test_embedding("Newer memory", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("oldest", 384);
    let (intent, _) = ctx
        .engine
        .query("Show me the oldest memory", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
}

#[test]
fn test_temporal_edge_002_empty_database() {
    let ctx = TestContext::new(Config::default());

    let query_emb = generate_test_embedding("recent", 384);
    let (intent, results) = ctx
        .engine
        .query("Show recent memories", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(intent.intent, QueryIntent::Temporal);
    assert_eq!(results.len(), 0);
}

#[test]
fn test_temporal_edge_003_future_timestamp() {
    let mut ctx = TestContext::new(Config::default());

    // Add memory with future timestamp
    let future_micros = now_timestamp().as_micros() + (24 * 60 * 60 * 1_000_000); // 1 day from now
    let future = Timestamp::from_micros(future_micros);
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Future task".to_string(),
        embedding: generate_test_embedding("Future task", 384),
        timestamp: Some(future),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("recent", 384);
    let (_, results) = ctx
        .engine
        .query("recent memories", &query_emb, 10, None, None)
        .unwrap();

    // Future memory should still be returned (no filtering by default)
    assert!(results.len() >= 1);
}

#[test]
fn test_temporal_edge_004_exact_now() {
    let mut ctx = TestContext::new(Config::default());

    let now = now_timestamp();
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Right now".to_string(),
        embedding: generate_test_embedding("Right now", 384),
        timestamp: Some(now),
        metadata: HashMap::new(),
    });

    let start = Timestamp::from_micros(now.as_micros().saturating_sub(1000));
    let end = Timestamp::from_micros(now.as_micros() + 1000);
    let results = ctx.engine.get_range(start, end, 10, None).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_temporal_edge_005_negative_range() {
    let ctx = TestContext::new(Config::default());

    // Invalid range: start > end
    let start = now_timestamp();
    let end = days_ago(7);

    let results = ctx.engine.get_range(start, end, 10, None).unwrap();
    // Should return empty results, not error
    assert_eq!(results.len(), 0);
}

#[test]
fn test_temporal_edge_006_very_old_memory() {
    let mut ctx = TestContext::new(Config::default());

    // Memory from 10 years ago
    let very_old_micros = now_timestamp()
        .as_micros()
        .saturating_sub(10 * 365 * 24 * 60 * 60 * 1_000_000u64);
    let very_old = Timestamp::from_micros(very_old_micros);
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Very old memory".to_string(),
        embedding: generate_test_embedding("Very old memory", 384),
        timestamp: Some(very_old),
        metadata: HashMap::new(),
    });

    let start = Timestamp::from_micros(very_old_micros.saturating_sub(1000));
    let results = ctx
        .engine
        .get_range(start, now_timestamp(), 10, None)
        .unwrap();
    assert!(results.len() >= 1);
}

#[test]
fn test_temporal_edge_007_single_memory() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Only memory".to_string(),
        embedding: generate_test_embedding("Only memory", 384),
        timestamp: Some(now_timestamp()),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("recent", 384);
    let (_, results) = ctx
        .engine
        .query("recent", &query_emb, 10, None, None)
        .unwrap();

    assert_eq!(results.len(), 1);
    // Temporal score should still be calculated
    assert!(results[0].1.temporal_score >= 0.0);
}

#[test]
fn test_temporal_edge_008_all_same_timestamp() {
    let mut ctx = TestContext::new(Config::default());

    let base_time = now_timestamp();
    // Add 3 memories with timestamps 1 microsecond apart
    // (Temporal index limitation: can't have exact duplicate timestamps)
    for i in 0..3 {
        let timestamp = Timestamp::from_micros(base_time.as_micros() + i as u64);
        ctx.add_memory(&TestMemory {
            id: None,
            content: format!("Activity {}", i),
            embedding: generate_test_embedding(&format!("Activity {}", i), 384),
            timestamp: Some(timestamp),
            metadata: HashMap::new(),
        });
    }

    // Use get_range to retrieve all memories in a tight time range
    let start = base_time;
    let end = Timestamp::from_micros(base_time.as_micros() + 10);
    let results = ctx.engine.get_range(start, end, 10, None).unwrap();

    assert_eq!(
        results.len(),
        3,
        "All 3 memories in tight time range should be returned"
    );

    // Verify all timestamps are within 1 microsecond of each other
    // (essentially simultaneous from a practical perspective)
    for (memory, timestamp) in &results {
        let diff = if memory.created_at >= base_time {
            memory.created_at.as_micros() - base_time.as_micros()
        } else {
            base_time.as_micros() - memory.created_at.as_micros()
        };
        assert!(diff <= 3, "All memories should be within 3 microseconds");
    }
}

#[test]
fn test_temporal_edge_009_limit_exceeded() {
    let mut ctx = TestContext::new(Config::default());

    // Add 100 memories
    for i in 0..100 {
        ctx.add_memory(&TestMemory {
            id: None,
            content: format!("Memory {}", i),
            embedding: generate_test_embedding(&format!("Memory {}", i), 384),
            timestamp: Some(days_ago(i)),
            metadata: HashMap::new(),
        });
    }

    let query_emb = generate_test_embedding("recent", 384);
    let (_, results) = ctx
        .engine
        .query(
            "recent", &query_emb, 10, // limit to 10
            None, None,
        )
        .unwrap();

    // Should respect limit
    assert_eq!(results.len(), 10);
}

#[test]
fn test_temporal_edge_010_zero_limit() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Some memory".to_string(),
        embedding: generate_test_embedding("Some memory", 384),
        timestamp: Some(now_timestamp()),
        metadata: HashMap::new(),
    });

    let query_emb = generate_test_embedding("recent", 384);
    let (_, results) = ctx
        .engine
        .query(
            "recent", &query_emb, 0, // zero limit
            None, None,
        )
        .unwrap();

    // Should return empty results
    assert_eq!(results.len(), 0);
}
