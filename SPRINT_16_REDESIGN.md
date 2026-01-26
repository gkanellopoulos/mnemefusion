# Sprint 16: Dimension Scoring Fundamentals Redesign

**Date**: January 26, 2026
**Status**: 📋 PLANNED
**Context**: Sprint 16.1 and 16.2 failed because they tried to fix fusion when the problem is dimension scoring itself

---

## Critical Discovery: Dimensions Measure The Wrong Things

### The Core Problem

**We're measuring METADATA when we should be measuring CONTENT.**

| Dimension | Currently Measures | Should Measure | Why Current Fails |
|-----------|-------------------|----------------|-------------------|
| **Temporal** | Timestamp metadata (recency) | Temporal expressions in content | "When did Alice go to the meeting?" doesn't care about recency |
| **Entity** | Capitalized words | Meaningful entities mentioned | "The Project" extracts "The" as entity |
| **Causal** | Nothing (returns 0s) | Causal language patterns | No signal = wasted weight |
| **Semantic** | ✅ Content embeddings | ✅ Content embeddings | **Only dimension measuring content correctly** |

### Why This Violates Core Value Proposition

From foundational documents:
> **"4D fusion mimics how human brain retrieves memories"**

**Human memory retrieval:**
- **Temporal**: "Remembers WHEN by associating events with time context" → Not just "newest memory"
- **Entity**: "Remembers WHO/WHAT was involved" → Not just "words that are capitalized"
- **Causal**: "Remembers WHY by understanding cause-effect" → Not nothing
- **Semantic**: "Remembers WHAT by understanding meaning" → ✅ We do this correctly

**Our current implementation:**
- Temporal = timestamp ordering (metadata-based)
- Entity = capitalization check (format-based)
- Causal = empty (not implemented)
- Semantic = embedding similarity (content-based) ✅

**Only 1 out of 4 dimensions measures content. This is why 4D fusion doesn't beat semantic-only.**

---

## Sprint 16 Redesign: Measure Content, Not Metadata

### Principle: Content-First Dimension Scoring

**Before each dimension scores a memory, ask:**
1. "What does the QUERY CONTENT say about this dimension?"
2. "What does the MEMORY CONTENT say about this dimension?"
3. "How well do they match?"

**NOT:**
- "When was this memory created?" (metadata)
- "Does this word start with capital letter?" (format)
- "What does the graph structure say?" (metadata)

---

## Sprint 16.1: Temporal Content Matching (~8 hours)

### Current Implementation (WRONG)
```rust
// temporal_search() returns memories sorted by timestamp
// Newest = 1.0, oldest = 0.0
// Query content IGNORED
```

### Redesigned Implementation (CORRECT)

**Step 1: Extract temporal expressions from query**
```rust
// Example queries:
// "What happened yesterday?" → [TemporalExpression::Yesterday]
// "Show me last week's meetings" → [TemporalExpression::LastWeek]
// "When did Alice go to the conference?" → [] (no time filter, just asking "when")
```

**Step 2: Extract temporal expressions from memory content (during ingestion)**
```rust
// Store in metadata:
// "We had a meeting yesterday about the project" →
//   temporal_expressions: ["yesterday", "about the project"]
//
// "The conference was on June 15th" →
//   temporal_expressions: ["June 15th", "was on"]
```

**Step 3: Score based on temporal content overlap**
```rust
fn temporal_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
    // Extract temporal expressions from query
    let query_temporal = extract_temporal_expressions(query_text);

    if query_temporal.is_empty() {
        // No temporal filter → use recency as weak fallback
        return Ok(recency_scores_with_low_weight());
    }

    // Score memories based on temporal content match
    let mut scores = HashMap::new();
    for memory in self.get_all_memories() {
        let memory_temporal = memory.metadata.get("temporal_expressions");
        let overlap_score = calculate_temporal_overlap(query_temporal, memory_temporal);
        if overlap_score > 0.0 {
            scores.insert(memory.id, overlap_score);
        }
    }

    Ok(scores)
}
```

**Key Insight:**
- Temporal dimension measures "does this memory discuss the same time period as the query?"
- NOT "is this memory recent?"

### Implementation Tasks

1. **Add temporal expression extraction** (~3 hours)
   - Use regex patterns for common temporal phrases
   - Extract during ingestion, store in metadata
   - Extract from query text during search

2. **Implement temporal content matching** (~3 hours)
   - Calculate overlap between query and memory temporal expressions
   - Score based on: exact match (1.0), partial match (0.5-0.8), no match (0.0)
   - Fall back to recency only when no temporal content exists

3. **Add tests** (~2 hours)
   - Test temporal extraction from various query types
   - Test scoring based on content overlap
   - Verify recency fallback for non-temporal queries

**Expected Impact:**
- Temporal queries like "What happened yesterday?" now find memories mentioning "yesterday" in content
- Non-temporal queries like "machine learning techniques" don't get biased by recency
- **Human-like behavior:** Match temporal context, not just timestamps

---

## Sprint 16.2: Entity Content Matching (~6 hours)

### Current Implementation (WRONG)
```rust
// entity_search() extracts capitalized words
// "The Project Alpha" → ["The", "Project", "Alpha"]
// High false positive rate, meaningless matches
```

### Redesigned Implementation (CORRECT)

**Step 1: Extract meaningful entities from query**
```rust
// Use noun phrase extraction + proper noun detection
// "Tell me about Alice and Project Alpha" →
//   entities: ["Alice", "Project Alpha"]
//
// NOT: ["Tell", "Alpha"] (capitalized but not entities)
```

**Step 2: Extract meaningful entities from memory content (during ingestion)**
```rust
// Store in metadata:
// "Alice presented Project Alpha at the conference" →
//   entities: ["Alice", "Project Alpha", "conference"]
//
// Simple approach: noun phrases + proper nouns
// Advanced (future): Named Entity Recognition (NER)
```

**Step 3: Score based on entity content overlap**
```rust
fn entity_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
    // Extract entities from query (simple: noun phrases, not stop words)
    let query_entities = extract_entities(query_text);

    if query_entities.is_empty() {
        // No entity focus → minimal entity scoring
        return Ok(HashMap::new());
    }

    // Score memories based on entity overlap
    let mut scores = HashMap::new();
    for memory in self.get_all_memories() {
        let memory_entities = memory.metadata.get("entities");
        let overlap_score = calculate_entity_overlap(query_entities, memory_entities);
        if overlap_score > 0.0 {
            scores.insert(memory.id, overlap_score);
        }
    }

    Ok(scores)
}
```

**Key Insight:**
- Entity dimension measures "does this memory mention the same entities as the query?"
- NOT "does this memory have capitalized words?"

### Implementation Tasks

1. **Add entity extraction** (~2 hours)
   - Extract noun phrases (not just capitalized words)
   - Filter stop words ("The", "A", "What", etc.)
   - Store during ingestion in metadata

2. **Implement entity content matching** (~2 hours)
   - Calculate overlap using set intersection
   - Score with TF-IDF weighting (common entities = lower weight)
   - Handle partial entity matches ("Project" matches "Project Alpha")

3. **Add tests** (~2 hours)
   - Test entity extraction filters stop words
   - Test scoring based on entity overlap
   - Verify no false positives from capitalization

**Expected Impact:**
- Entity queries like "Show me memories about Alice" find memories actually mentioning Alice
- Queries without entities don't get noise from random capitalized words
- **Human-like behavior:** Match actual entities, not just formatting

---

## Sprint 16.3: Causal Language Scoring (~6 hours)

### Current Implementation (WRONG)
```rust
// causal_search() returns empty HashMap
// No signal, wasted weight
```

### Redesigned Implementation (CORRECT)

**Step 1: Detect causal language in query**
```rust
// "Why did the meeting get cancelled?" → HAS_CAUSAL_INTENT
// "What caused the server crash?" → HAS_CAUSAL_INTENT
// "Tell me about machine learning" → NO_CAUSAL_INTENT
```

**Step 2: Detect causal language in memory content (during ingestion)**
```rust
// Store in metadata:
// "The meeting was cancelled because Alice was sick" →
//   causal_expressions: ["because", "was cancelled"]
//   causal_density: 0.15 (15% of words are causal markers)
//
// "We had a nice lunch today" →
//   causal_expressions: []
//   causal_density: 0.0
```

**Step 3: Score based on causal language presence**
```rust
fn causal_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
    // Check if query has causal intent
    let has_causal_intent = query_contains_causal_keywords(query_text);

    if !has_causal_intent {
        // No causal focus → minimal causal scoring
        return Ok(HashMap::new());
    }

    // Score memories based on causal language density
    let mut scores = HashMap::new();
    for memory in self.get_all_memories() {
        let causal_density = memory.metadata.get("causal_density").unwrap_or(0.0);
        if causal_density > 0.1 {
            scores.insert(memory.id, causal_density);
        }
    }

    // Optionally: boost scores using causal graph when available
    if let Some(graph) = self.causal_graph.as_ref() {
        for (id, score) in scores.iter_mut() {
            if graph.has_causal_links(id) {
                *score *= 1.2; // Boost memories with explicit causal relationships
            }
        }
    }

    Ok(scores)
}
```

**Key Insight:**
- Causal dimension measures "does this memory explain causes/effects?"
- NOT "is this memory connected in the causal graph?" (graph is supplementary)
- Language-first, graph-second approach

### Implementation Tasks

1. **Add causal language detection** (~2 hours)
   - Regex patterns for causal markers: "because", "caused", "led to", "resulted in", "due to", "reason"
   - Calculate causal density during ingestion
   - Store in metadata

2. **Implement causal scoring** (~2 hours)
   - Score based on causal language density
   - Optionally boost with causal graph (if available)
   - Return empty for non-causal queries

3. **Add tests** (~2 hours)
   - Test causal language detection
   - Test scoring prioritizes causal-rich content
   - Verify no false positives

**Expected Impact:**
- "Why" queries find memories that actually explain causes
- Non-causal queries don't get noise from arbitrary graph connections
- **Human-like behavior:** Match causal reasoning in content

---

## Sprint 16.4: Validate 4D Fusion (~2 hours)

### Objective
Verify that content-based dimension scoring makes 4D fusion better than semantic-only.

### Tasks

1. **Re-run LoCoMo Phase 2** (~1 hour)
   - Run full benchmark with redesigned dimensions
   - Compare to semantic-only baseline (38.5%)
   - Expected: **Significant improvement** (45-55%+)

2. **Analyze dimension contributions** (~30 min)
   - Check which dimensions contributed to which queries
   - Verify dimensions provide meaningful signals
   - Document dimension effectiveness

3. **Update documentation** (~30 min)
   - Document new dimension scoring approach
   - Update PROJECT_STATE.md with results
   - Update IMPLEMENTATION_PLAN.md status

### Success Criteria

**Must Achieve:**
- ✅ LoCoMo Phase 2 > 45% (beating semantic-only by 6.5%+)
- ✅ Each dimension contributes positively to its target queries
- ✅ No regression on non-target queries

**Would Validate Core Value Prop:**
- ✅ 4D fusion demonstrably better than semantic-only
- ✅ Intent-aware query planning provides value
- ✅ System behaves more like human memory retrieval

---

## Sprint 16 Summary

### Total Time: ~22 hours (fits in 2-week sprint)

| Task | Hours | Priority | Impact |
|------|-------|----------|--------|
| Sprint 16.1: Temporal Content | 8 | HIGH | Core dimension fix |
| Sprint 16.2: Entity Content | 6 | HIGH | Core dimension fix |
| Sprint 16.3: Causal Language | 6 | MEDIUM | Enable causal queries |
| Sprint 16.4: Validation | 2 | HIGH | Prove value prop |

### Expected Results

**Before Sprint 16** (Current Baseline):
- LoCoMo Phase 2: 38.5%
- 4D fusion = semantic-only (dimensions add no value)
- Core value prop not validated

**After Sprint 16** (Expected):
- LoCoMo Phase 2: **48-55%** (+9.5 to +16.5 percentage points)
- 4D fusion > semantic-only (dimensions measurably help)
- Core value prop validated ✅

### Why This Approach Is Correct

**Aligns with Core Value Proposition:**
- "Mimics how human brain retrieves memories" ✅
- Measures CONTENT (what memories are about)
- Not METADATA (when stored, how formatted)

**Works Broadly (Not Dataset-Specific):**
- Temporal: Works for any content with time references
- Entity: Works for any content with meaningful entities
- Causal: Works for any content with explanations
- Not tuned for LoCoMo specifically

**Testable & Validatable:**
- Each dimension can be tested independently
- Clear success criteria (beat semantic-only)
- Measurable impact per dimension

**Maintainable & Extensible:**
- Simple regex-based extraction (can be improved later)
- Content stored in metadata (no schema changes)
- Foundation for future ML-based extraction

---

## Migration Path

### Phase 1: Ingestion Changes (Required First)
1. Update ingestion pipeline to extract:
   - Temporal expressions from content
   - Meaningful entities from content
   - Causal language markers from content
2. Store in metadata (backward compatible)
3. Re-index existing data (migration script)

### Phase 2: Query Changes
1. Update temporal_search() to use content matching
2. Update entity_search() to use content matching
3. Update causal_search() to use language density
4. Keep fusion engine unchanged (weights already correct)

### Phase 3: Validation
1. Run benchmarks
2. Verify improvement
3. Document results

---

## Risk Assessment

**LOW RISK:**
- ✅ No breaking API changes (internal implementation only)
- ✅ Backward compatible (metadata-based storage)
- ✅ Can roll back if needed (keep old code paths)
- ✅ Incremental (can deploy one dimension at a time)

**MEDIUM COMPLEXITY:**
- ⚠️ Requires re-indexing existing data
- ⚠️ More computation during ingestion
- ⚠️ Regex patterns may need tuning

**HIGH IMPACT:**
- 🎯 Validates core value proposition
- 🎯 Makes 4D fusion actually useful
- 🎯 Foundation for future ML-based improvements

---

## Next Steps After Sprint 16

**If Sprint 16 Succeeds (Expected):**
1. Sprint 17: Improve extraction quality (ML-based NER, temporal parsing)
2. Sprint 18: Optimize performance (caching, indexing)
3. Sprint 19: API documentation and 1.0 release

**If Sprint 16 Fails (Unexpected):**
1. Analyze which dimensions didn't help and why
2. Consider alternative scoring strategies
3. Re-evaluate core value proposition

---

**Status**: 📋 READY TO IMPLEMENT
**Confidence**: HIGH (fixes fundamental issues, not fusion tweaks)
**Expected Outcome**: 4D fusion beats semantic-only by 10-15%+

---
