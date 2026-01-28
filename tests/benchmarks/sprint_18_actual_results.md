# Sprint 18 Actual Results: Analysis of Minimal Impact

**Date**: 2026-01-28
**Status**: ⚠️ Below Expectations

## Executive Summary

Sprint 18 implemented three major improvements (precision, entity-first retrieval, temporal content matching) but benchmarks show **virtually no improvement** over Sprint 17. This document analyzes why the improvements didn't work as expected.

---

## Benchmark Results Comparison

### LoCoMo Phase 2 (1,986 queries)

| Metric | Sprint 17 | Sprint 18 | Change | Expected |
|--------|-----------|-----------|---------|----------|
| **Overall Recall@10** | 65.05% | 65.14% | **+0.09%** | +3-7% ❌ |
| **Precision@10** | 7.29% | 7.30% | **+0.01%** | +10% ❌ |
| **MRR** | 0.241 | 0.238 | **-0.003** | +5% ❌ |

### Category Breakdown

| Category | Sprint 17 | Sprint 18 | Change | Expected |
|----------|-----------|-----------|---------|----------|
| **Category 1** (Entity) | 16.18% | 16.07% | **-0.11%** | +24-34% ❌ |
| Category 2 (Recent Temporal) | 72.14% | 72.46% | +0.32% | +0-2% ✅ |
| **Category 3** (Specific Temporal) | 23.18% | 22.14% | **-1.04%** | +12-22% ❌ |
| Category 4 (Conversation Flow) | 74.08% | 74.44% | +0.36% | +1-3% ✅ |
| Category 5 (Session Boundaries) | 82.85% | 82.62% | -0.23% | +0-2% ~ |

### LongMemEval Oracle (500 queries)

| Metric | Sprint 17 | Sprint 18 | Change | Expected |
|--------|-----------|-----------|---------|----------|
| **Recall@10** | 70.1% | 71.0% | **+0.9%** | +3-5% ⚠️ |
| **Precision@10** | 15.4% | 15.5% | **+0.1%** | +10% ❌ |
| **F1 Score** | 25.6% | 24.8% | **-0.8%** | +5% ❌ |

---

## Root Cause Analysis

### Problem 1: Intent Classification Doesn't Match LoCoMo Queries

**The Core Issue**: Our improvements rely on correct intent classification, but LoCoMo queries are classified as `Factual` (default semantic search) instead of `Entity` or `Temporal`.

#### Category 1 (Entity Queries) - Intent Distribution

| Intent | Percentage | Impact |
|--------|------------|--------|
| **Factual** | **81.6%** | Uses standard semantic search (no entity-first retrieval) |
| Entity | 10.3% | Only 10% get entity-first retrieval ❌ |
| Temporal | 6.7% | Wrong intent |
| Causal | 1.4% | Wrong intent |

**Example Mismatches**:
```
Query: "What hobbies does Alice have?"
Our Pattern: "What does Alice like/enjoy/prefer?"
Result: ❌ Doesn't match → classified as Factual

Query: "List all things about the project"
Our Pattern: "List all about Project"
Result: ❌ "things about" vs "about" → classified as Factual
```

**Impact**: Only **10.3%** of Category 1 queries get entity-first retrieval, so **89.7%** fall back to standard semantic search (which is why recall didn't improve).

#### Category 3 (Temporal Queries) - Intent Distribution

| Intent | Percentage | Impact |
|--------|------------|--------|
| **Factual** | **74.0%** | Uses standard semantic search (no temporal content matching) |
| Temporal | 10.4% | Only 10% get temporal content matching ❌ |
| Entity | 10.4% | Wrong intent |
| Causal | 5.2% | Wrong intent |

**Example Mismatches**:
```
Query: "When did we discuss the budget?"
Our Pattern: "when", "before", "after"
Result: ⚠️ Matches but weak score → classified as Factual

Query: "What happened on Monday?"
Our Pattern: "Monday" (weekday name)
Result: ⚠️ Matches but weak score → classified as Factual
```

**Impact**: Only **10.4%** of Category 3 queries get temporal content matching, so **89.6%** use weak recency fallback.

---

### Problem 2: Precision Improvements Not Effective

**Expected**: Semantic prefilter (0.3) + cross-dimensional validation → precision 15.4% → 25-28%

**Actual**: Precision 15.4% → 15.5% (+0.1%)

**Why**:

1. **Semantic Prefilter Too Aggressive**: Threshold 0.3 may filter out relevant results
2. **Cross-Dimensional Validation Rarely Helps**: Most results are single-dimension (semantic only)
3. **LoCoMo Precision Already Low**: 7.3% precision is measuring a different problem (relevance judgment)

**Evidence**:
- LoCoMo precision stayed at ~7.3% (unchanged)
- LongMemEval precision 15.4% → 15.5% (minimal)
- The improvements filtered some results but didn't improve what got through

---

### Problem 3: Implementations Are Correct But Underutilized

**Key Insight**: The implementations work correctly when triggered, but they're **not being triggered** due to intent classification failures.

#### Entity-First Retrieval Test Results ✅

```rust
#[test]
fn test_entity_focused_retrieval() {
    // Query: "What does Alice like?"
    let (intent, results) = planner.query(...).unwrap();

    assert_eq!(intent.intent, QueryIntent::Entity); // ✅ Passes
    assert_eq!(intent.entity_focus, Some("Alice")); // ✅ Passes
    assert!(results.len() >= 3); // ✅ Fetches all Alice memories
}
```

**Unit tests pass**, proving the implementation works when the query matches our patterns.

#### Temporal Content Matching Test Results ✅

```rust
#[test]
fn test_search_temporal_content_basic() {
    // Query: "What happened yesterday?"
    let results = temporal.search_temporal_content(...).unwrap();

    assert!(!results.is_empty()); // ✅ Passes
    assert_eq!(results[0].0, mem1.id); // ✅ Finds "yesterday" match
}
```

**Unit tests pass**, proving temporal content matching works when temporal expressions are extracted.

---

## Why Did This Happen?

### 1. Pattern Coverage Gap

**Our Patterns**: Designed for simple, direct queries
```
- "What does Alice like?"
- "List all about Project"
- "What happened yesterday?"
```

**LoCoMo Queries**: More varied and complex
```
- "What hobbies does Alice have?" (different phrasing)
- "Tell me about Alice's interests" (possessive form variation)
- "When did we talk about X?" (indirect temporal)
```

**Solution Needed**: More comprehensive regex patterns or fuzzy matching.

### 2. Intent Classification Scoring Too Conservative

**Current Scoring**:
```rust
// Temporal patterns
if temporal_matches > 0 {
    score = (matches as f32 * 0.4).min(1.0);
}

// Entity list patterns (strong indicators)
if entity_list_matches > 0 {
    score += (matches as f32 * 0.6).min(1.0);
}

// Factual baseline: 0.3 (always present)
```

**Problem**: `Factual` gets base score of 0.3, so single weak match (0.4) often loses to `Factual` when considering secondary signals.

**Example**:
```
Query: "When did we meet?"
- Temporal match: "when" → score 0.4
- Factual baseline: 0.3
- Entity pattern: "we" (capitalized) → 0.2
- Total Entity: 0.5 > Temporal: 0.4
- Result: Classified as Entity (wrong!) instead of Temporal
```

### 3. Semantic Search Is Already Strong

**Insight**: For many queries, semantic search alone gets ~65% recall. The remaining 35% are the truly hard cases that require:
- Complex multi-hop reasoning
- Exact temporal context matching
- Aggregation across 20+ turns

**Our improvements help with these cases**, but since only 10-20% of queries are correctly classified to use them, the overall impact is minimal.

---

## What Worked vs What Didn't

### ✅ What Worked (Implementation Quality)

1. **Clean Code**: All 345 tests passing, well-structured
2. **Correct Logic**: Entity-first retrieval works when triggered
3. **Temporal Matching**: Temporal content scoring works when triggered
4. **No Regressions**: Didn't break existing functionality

### ❌ What Didn't Work (Real-World Impact)

1. **Intent Classification Coverage**: Only 10% of target queries classified correctly
2. **Pattern Matching**: Too specific, misses variations
3. **Precision Improvements**: Minimal impact on benchmark precision
4. **Overall Recall**: No meaningful improvement (65.05% → 65.14%)

---

## Lessons Learned

### 1. Test on Real Data Early

**Mistake**: Designed patterns based on intuition, tested with synthetic examples
**Better**: Analyze actual LoCoMo queries first, design patterns to match them

### 2. Intent Classification Is Critical

**Mistake**: Assumed simple regex patterns would catch most queries
**Better**: Use more comprehensive patterns OR use embedding-based intent classification

### 3. Precision vs Recall Trade-off

**Mistake**: Focused on precision improvements when recall is the bottleneck
**Better**: LoCoMo cares about recall (finding the needle). Precision (7.3%) is measuring something else (relevance).

### 4. Benchmark Variability

**Mistake**: Expected large improvements (+24-34%) from targeted changes
**Better**: In mature systems, even good improvements yield small gains (1-3%)

---

## Path Forward: Options Analysis

### Option A: Improve Intent Classification (Recommended)

**Approach**: Expand patterns to cover more query variations

**Changes Needed**:
```rust
// Current: Too specific
Regex::new(r"^what\s+does\s+(\w+)\s+(like|enjoy|prefer)")

// Better: More flexible
Regex::new(r"(?i)\b(?:what|which)\s+.*?(\w+)'?s?\s+.*(hobby|hobbies|interest|like|prefer|enjoy)")
```

**Expected Impact**: Category 1/3 intent detection: 10% → 40-50%
**Effort**: 2-4 hours (pattern expansion + testing)
**Risk**: Low (adding patterns, not changing logic)

### Option B: Embedding-Based Intent Classification

**Approach**: Train simple classifier on query embeddings

**Changes Needed**:
- Collect labeled examples (500-1000 queries)
- Train logistic regression on query embeddings
- Replace regex with model prediction

**Expected Impact**: Category 1/3 intent detection: 10% → 60-70%
**Effort**: 1-2 days (data collection + training + integration)
**Risk**: Medium (adds dependency, needs training data)

### Option C: Accept Current Performance

**Rationale**:
- 65% recall is still competitive for zero-dependency solution
- LoCoMo is artificially hard (multi-turn aggregation problem)
- Real users may have simpler queries

**Next Steps**:
- Document current limitations
- Move to productionization
- Gather real user feedback

**Risk**: May not meet "SQLite for AI memory" threshold (68-70%)

### Option D: Hybrid Approach (Entity Extraction First)

**Approach**: Always extract entities and use them to boost retrieval, regardless of intent

**Changes Needed**:
```rust
// In query():
let entities = extract_entities_from_query(query_text);
if !entities.is_empty() {
    // Boost entity scores for all results
    for entity in entities {
        boost_scores_for_entity(&mut all_scores, entity);
    }
}
```

**Expected Impact**: Category 1: 16% → 25-30% (partial improvement)
**Effort**: 4-6 hours
**Risk**: Low (additive, doesn't break existing paths)

---

## Recommendation

### Immediate (Next 4-8 hours)

**Expand intent classification patterns** (Option A):
1. Analyze failed Category 1 & 3 queries from benchmark
2. Add 10-15 more comprehensive patterns
3. Re-run benchmarks
4. Target: Category 1: 16% → 25-30%, Category 3: 22% → 30-35%

### Short-term (Next 1-2 days)

**Hybrid entity boosting** (Option D):
1. Extract entities from ALL queries (not just Entity intent)
2. Boost scores for memories mentioning those entities
3. Provides partial benefit without perfect intent classification
4. Target: Overall recall: 65% → 67-68%

### Long-term (Future Sprint)

**Embedding-based intent classification** (Option B):
- Build dataset from benchmark queries
- Train classifier
- Replace regex patterns
- Target: 70%+ recall

---

## Conclusion

Sprint 18 implementations are **technically correct** but have **minimal real-world impact** due to:

1. **Intent Classification Gap**: Only 10% of target queries trigger improvements
2. **Pattern Specificity**: Regex patterns too narrow for query variations
3. **Precision Trade-off**: Focused on precision when recall is the bottleneck

**Current Status**: 65.14% recall (target was 68-70%) ❌

**Path Forward**:
- **Quick fix**: Expand patterns → ~67-68% recall
- **Better fix**: Hybrid entity boosting → ~68-70% recall
- **Best fix**: Embedding-based intent → ~70%+ recall

**Decision Point**: Accept 65%, quick fix to 68%, or invest in longer-term solution?
