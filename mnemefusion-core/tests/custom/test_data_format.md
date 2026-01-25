# Custom Test Data Format

This document defines the JSON format for custom test fixtures used to validate MnemeFusion's differentiators (temporal, causal, entity, intent).

## Test Case Structure

Each test case has the following structure:

```json
{
  "id": "temporal_001",
  "name": "Recent memories query",
  "category": "temporal",
  "query": {
    "text": "What happened recently?",
    "embedding": [0.1, 0.2, ...],  // Optional: pre-computed embedding
    "expected_intent": "temporal",
    "expected_confidence": ">0.4"
  },
  "setup": {
    "memories": [
      {
        "content": "Meeting with Alice yesterday",
        "timestamp": "2026-01-23T10:00:00Z",
        "embedding": [0.15, 0.25, ...],
        "metadata": {"type": "event"}
      },
      {
        "content": "Project deadline moved to March",
        "timestamp": "2026-01-20T14:00:00Z",
        "embedding": [0.12, 0.22, ...],
        "metadata": {"type": "task"}
      }
    ],
    "causal_links": [],  // For causal tests
    "entities": []       // For entity tests
  },
  "expected": {
    "intent": "temporal",
    "intent_confidence_min": 0.4,
    "results_count_min": 1,
    "results_must_include": ["Meeting with Alice yesterday"],
    "temporal_score_threshold": 0.5,
    "fusion_weights": {
      "temporal_min": 0.4,
      "semantic_max": 0.4
    },
    "time_range": {
      "relative": "last_7_days",
      "absolute_start": null,
      "absolute_end": null
    },
    "ordering": "newest_first"
  },
  "validation": {
    "assert_intent": true,
    "assert_results_in_timerange": true,
    "assert_fusion_weights": true,
    "assert_ordering": true
  }
}
```

## Category-Specific Fields

### Temporal Tests

```json
{
  "expected": {
    "time_range": {
      "relative": "last_24h" | "last_7_days" | "last_month",
      "absolute_start": "2026-01-01T00:00:00Z",
      "absolute_end": "2026-01-31T23:59:59Z"
    },
    "ordering": "newest_first" | "oldest_first" | "relevance"
  }
}
```

### Causal Tests

```json
{
  "setup": {
    "causal_links": [
      {
        "from_content": "Storm forecast for tomorrow",
        "to_content": "Meeting cancelled",
        "confidence": 0.9
      }
    ]
  },
  "expected": {
    "causal_chain": [
      "Storm forecast for tomorrow",
      "Meeting cancelled"
    ],
    "max_hops": 2,
    "min_confidence": 0.5
  }
}
```

### Entity Tests

```json
{
  "setup": {
    "entities": [
      {
        "name": "Alice",
        "mentions_in": ["Meeting with Alice yesterday", "Alice completed Project Alpha"]
      },
      {
        "name": "Project Alpha",
        "mentions_in": ["Alice completed Project Alpha"]
      }
    ]
  },
  "expected": {
    "entities_detected": ["Alice"],
    "entity_memories_count_min": 2
  }
}
```

### Intent Classification Tests

```json
{
  "expected": {
    "primary_intent": "factual",
    "secondary_intents": [],
    "intent_confidence_min": 0.3,
    "mixed_intent": false
  }
}
```

### Fusion Tests

```json
{
  "baseline": {
    "semantic_only_recall": 0.6
  },
  "expected": {
    "fusion_recall_min": 0.7,
    "improvement_over_baseline": ">10%",
    "weights_sum_to_one": true,
    "weights": {
      "semantic": 0.3,
      "temporal": 0.5,
      "causal": 0.1,
      "entity": 0.1
    }
  }
}
```

## Simplified Format for Common Cases

For simple tests, a minimal format is supported:

```json
{
  "id": "temporal_002",
  "category": "temporal",
  "query": "yesterday",
  "setup": ["Memory from yesterday", "Memory from last week"],
  "expected_intent": "temporal",
  "expected_results": ["Memory from yesterday"]
}
```

This will be expanded internally to the full format with defaults.

## Test Fixture Files

Test fixtures are organized by category:

- `temporal_tests.json` - 50 temporal test cases
- `causal_tests.json` - 60 causal test cases
- `entity_tests.json` - 35 entity test cases
- `intent_tests.json` - 25 intent classification test cases
- `fusion_tests.json` - 10 fusion validation test cases

## Embedding Generation

Test cases can either:
1. Provide pre-computed embeddings in the JSON
2. Leave embeddings empty and generate them at runtime using a standard model (all-MiniLM-L6-v2)

For reproducibility, we recommend pre-computing embeddings and storing them in the fixtures.

## Assertion Types

The validation framework supports these assertion types:

- `assert_intent` - Validates intent classification
- `assert_results_in_timerange` - Validates temporal range filtering
- `assert_fusion_weights` - Validates fusion weight adaptation
- `assert_ordering` - Validates result ordering
- `assert_causal_chain` - Validates causal graph traversal
- `assert_entities_detected` - Validates entity extraction
- `assert_recall_improvement` - Validates fusion > baseline

## Example: Complete Temporal Test

```json
{
  "id": "temporal_recency_001",
  "name": "Query for recent memories returns newest first",
  "category": "temporal",
  "query": {
    "text": "What happened recently?",
    "expected_intent": "temporal"
  },
  "setup": {
    "memories": [
      {
        "id": "mem_1",
        "content": "Meeting with Alice this morning",
        "timestamp": "2026-01-24T09:00:00Z"
      },
      {
        "id": "mem_2",
        "content": "Lunch with Bob yesterday",
        "timestamp": "2026-01-23T12:00:00Z"
      },
      {
        "id": "mem_3",
        "content": "Project kickoff last week",
        "timestamp": "2026-01-17T14:00:00Z"
      }
    ]
  },
  "expected": {
    "intent": "temporal",
    "results_ordered": ["mem_1", "mem_2", "mem_3"],
    "temporal_score_decreasing": true,
    "fusion_weights": {
      "temporal_min": 0.4
    }
  },
  "validation": {
    "assert_intent": true,
    "assert_ordering": true,
    "assert_fusion_weights": true
  }
}
```
