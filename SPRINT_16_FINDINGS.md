# Sprint 16.1: Temporal Relevance - FAILED (Reverted)

**Date**: January 26, 2026
**Duration**: 3 hours (research + implementation + testing)
**Status**: ❌ **REVERTED** - Wrong approach for LoCoMo dataset

---

## Summary

Sprint 16.1 attempted to improve temporal queries by extracting time ranges ("yesterday", "last week") and searching those specific periods. This approach failed because **LoCoMo temporal queries ask "When did X happen?" (expecting dates as answers), not "Show me what happened yesterday" (filtering by time)**.

---

## What We Tried

### Implementation
1. Added `extract_temporal_range()` to `IntentClassifier` - parses temporal keywords
2. Modified `temporal_search()` to extract ranges and query those periods
3. Added `Timestamp` helper methods (`add_hours()`, `weekday()`, `start_of_month()`)
4. Reduced fallback recency scoring from 1.0 → 0.5 max to avoid overpowering semantic

### Expected Impact
- Category 2 (temporal): 43.0% → 60-70% (+10-15%)
- Overall: 38.5% → 50%+

### Actual Results
- Category 2 (temporal): 43.0% → **43.0%** (0% change)
- Category 4 (entity): 43.1% → **42.1%** (-1.0%)
- Category 5 (contextual): 49.1% → **47.1%** (-2.0%)
- **Overall: 38.5% → 37.7%** (-0.9% ❌)

---

## Root Cause Analysis

### The Misunderstanding

**LoCoMo Category 2 queries** (labeled "temporal") are NOT about filtering by time. They are questions where **the answer IS a time**:

```
Examples:
- "When did Caroline go to the LGBTQ support group?"  → Answer: "7 May 2023"
- "When did Melanie paint a sunrise?"                → Answer: specific date
- "How long has Caroline had her current group of friends for?"  → Answer: duration
```

These queries:
- **Do not contain** temporal filter keywords ("yesterday", "last week", "recent")
- **Do contain** "When" as a question word (asking for time as output)
- **Need semantic search** to find the right event
- **Temporal dimension should help** by scoring memories with explicit time references higher

### What We Implemented (WRONG)

```rust
// Extract temporal range from query
if query.contains("yesterday") {
    // Search memories from yesterday with 1.0 relevance
    return temporal_index.range_query(yesterday_start, yesterday_end, limit);
}

// Fallback to recency
return temporal_index.recent(limit);  // 0.5 max scores (reduced)
```

**Problem**: LoCoMo queries don't have "yesterday" keywords, so they always hit the fallback path with reduced (0.5 max) scores, which hurt performance.

### Why It Hurt Performance

1. **Temporal Intent Detection Works**: 85% of Category 2 queries correctly identified as Temporal intent
2. **Adaptive Weights Applied**: 35% weight to temporal dimension (down from 50% semantic)
3. **But Signal Weakened**: Temporal scores reduced from 1.0 → 0.5 max
4. **Effective Contribution Cut**: 35% × 0.5 = 17.5% vs 35% × 1.0 = 35%
5. **Semantic Got Relatively Stronger**: 50% semantic now dominates even more

---

## What "Temporal Relevance" Actually Means for LoCoMo

### For "When did X happen?" queries:

1. **Semantic search** finds memories about event X (primary signal)
2. **Temporal dimension** should boost memories that:
   - Contain explicit timestamps in metadata
   - Have temporal context in the text ("on May 7th", "during the summer")
   - Are clustered in time with other relevant events
3. **Recency bias is irrelevant** - we care about the right event, not recent events

### The Correct Approach (for future):

**Option A: Temporal Context Scoring**
- Score memories higher if they contain temporal phrases in text
- Use NER to extract dates/times from memory content
- Boost memories where timestamp matches other contextual clues

**Option B: Semantic-Only for "When" Questions**
- Detect "When did..." pattern
- Boost semantic weight to 80%, reduce temporal to 10%
- Let semantic search find the right event
- Temporal dimension adds minimal signal (timestamp metadata)

**Option C: Temporal Clustering**
- Find semantically similar memories
- Cluster them by timestamp
- Return memories from the densest temporal cluster
- Assumption: important events have multiple related memories

---

## Lessons Learned

### 1. Understand the Dataset Before Optimizing

- LoCoMo "temporal queries" ≠ "queries about recent events"
- They're questions where **time is the answer**, not the filter
- We assumed temporal meant "show me yesterday's memories"
- Actually means "tell me when X happened"

### 2. Recency Bias is Not Universal

- Recency makes sense for personal memory ("What did I do yesterday?")
- Doesn't help for conversational memory ("When did character X do Y?")
- LoCoMo is conversational, not personal
- The "right" answer could be from any time in the conversation history

### 3. Intent Classification vs Dimension Scoring

- Intent classification WORKS (85% accuracy for Category 2)
- But dimension scoring needs to match the intent type
- "Temporal intent" doesn't always mean "search by time"
- Sometimes means "find memories where time is relevant information"

### 4. Evaluation-Driven Development is Critical

- Quick Phase 1 validation (199 queries) showed +24.6% improvement
- Full Phase 2 (1986 queries) showed 0% improvement
- **Lesson**: Always validate on full dataset before celebrating

---

## Impact on Sprint 16 Plan

### Original Sprint 16 Tasks:
1. ✅ Task 16.1: Temporal Relevance (5 hours) - **DONE (but reverted)**
2. ⏳ Task 16.2: Signal Quality Detection (5.5 hours)
3. ⏳ Task 16.3: Improve Entity Extraction (3 hours)
4. ⏳ Task 16.4: Implement Causal Search (5.5 hours)

### Revised Sprint 16 Tasks:
1. ~~Task 16.1: Temporal Relevance~~ - **CANCELLED** (wrong approach)
2. **Task 16.1b: Temporal Context Detection** (6 hours) - NEW
   - Detect "When" question pattern
   - Boost semantic for "When" queries
   - Add temporal phrase detection in content
3. Task 16.2: Signal Quality Detection (5.5 hours) - **UNCHANGED**
4. Task 16.3: Improve Entity Extraction (3 hours) - **UNCHANGED**
5. Task 16.4: Implement Causal Search (5.5 hours) - **UNCHANGED**

---

## Next Steps

### Immediate:
1. ✅ Revert Sprint 16.1 changes to restore baseline
2. ✅ Document findings in SPRINT_16_FINDINGS.md
3. ⏳ Update IMPLEMENTATION_PLAN.md with revised Sprint 16 plan
4. ⏳ Commit Sprint 15.5 changes (bug fixes + weight optimization)

### Sprint 16 Revised Plan:
1. **Focus on signal quality first** (Task 16.2) - detect weak signals and adapt
2. **Then entity extraction** (Task 16.3) - reduce false positives
3. **Then temporal context** (Task 16.1b) - proper "When" query handling
4. **Finally causal search** (Task 16.4) - basic traversal

### Why This Order:
- **Signal quality** (16.2) is foundational - helps all dimensions
- **Entity** (16.3) has clear false positive problem (capitalized words)
- **Temporal context** (16.1b) now informed by this failure
- **Causal** (16.4) is lowest priority (only ~5% of queries)

---

## Code Artifacts (for Reference)

### Files Changed (Reverted):
- `mnemefusion-core/src/query/intent.rs` - Added extract_temporal_range()
- `mnemefusion-core/src/query/planner.rs` - Modified temporal_search()
- `mnemefusion-core/src/types/timestamp.rs` - Added helper methods

### Tests Added (Kept):
- 6 temporal extraction tests in intent.rs (useful for future)
- Tests validate range parsing logic (correct, just wrong use case)

### Benchmark Results:
- `tests/benchmarks/fixtures/locomo_phase2_temporal_relevance.json` - 37.7% recall

---

## Retrospective

### What Went Well ✅
- **Fast iteration**: Implemented + tested in 3 hours
- **Proper evaluation**: Caught the issue with full Phase 2 run
- **Clean revert**: Changes isolated, easy to undo
- **Learning captured**: Documented for future reference

### What Went Wrong ❌
- **Insufficient research**: Didn't examine actual LoCoMo queries first
- **Assumption-driven**: Assumed "temporal" meant time-based filtering
- **Premature optimization**: Started coding before understanding problem
- **Ignored semantics**: Focused on temporal dimension, forgot semantic is primary

### What to Do Differently 🔄
- **Examine queries first**: Look at actual examples before designing solution
- **Question assumptions**: "Temporal query" can mean different things
- **Prototype quickly**: Could have tested extraction on 10 queries before implementing
- **Semantic-first**: Remember semantic search is the foundation, dimensions assist

---

**Status**: ❌ **REVERTED** - Changes undone, baseline restored
**Time Lost**: 3 hours (research + implementation + testing)
**Value Gained**: Deep understanding of LoCoMo dataset + temporal dimension semantics
**Next Priority**: Sprint 16.2 - Signal Quality Detection (foundational improvement)

---
