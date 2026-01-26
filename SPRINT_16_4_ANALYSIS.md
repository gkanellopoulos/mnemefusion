# Sprint 16.4: Validation Results - REGRESSION DETECTED ❌

**Date**: January 26, 2026
**Status**: ❌ REGRESSION - Need Investigation
**Time**: ~2 hours

---

## Summary

After implementing all three content-based dimension improvements (Sprint 16.1-16.3), re-running LoCoMo Phase 2 shows **significant regression** instead of improvement.

## Results

### Overall Performance

| Metric | Baseline (Semantic-Only) | After 4D Content Fixes | Change |
|--------|--------------------------|------------------------|--------|
| Recall@10 | **38.5%** | **27.3%** | **-11.2%** (-29.1%) |
| MRR | 0.203 | 0.146 | -0.057 (-28.1%) |
| Precision@10 | 0.043 | 0.032 | -0.011 (-25.6%) |

**Verdict**: MASSIVE REGRESSION across all metrics

### Performance by Category

| Category | Description | Baseline | After Fixes | Change |
|----------|-------------|----------|-------------|--------|
| 1 | Single-hop factual | 12.1% | 9.7% | -2.4% |
| 2 | Multi-hop reasoning | **43.0%** | **12.4%** | **-30.6%** ❌ |
| 3 | Temporal reasoning | 12.3% | 10.8% | -1.5% |
| 4 | Entity relationships | 43.1% | 34.7% | -8.4% |
| 5 | Contextual | 49.1% | 38.8% | -10.3% |

**Biggest regression**: Category 2 (Multi-hop reasoning) dropped by 30.6%!

### Intent Distribution Analysis

| Category | Top Intent (%) | Expected Intent | Match? |
|----------|----------------|-----------------|--------|
| 1 (Single-hop) | Factual (81.6%) | Factual | ✅ CORRECT |
| 2 (Multi-hop) | **Temporal (83.8%)** | Factual/Mixed | ❌ WRONG |
| 3 (Temporal) | **Factual (74.0%)** | Temporal | ❌ WRONG |
| 4 (Entity) | Factual (53.3%) | Entity | ⚠️ MIXED |
| 5 (Contextual) | Factual (56.7%) | Mixed | ✅ OK |

**Critical issues**:
1. Category 2 (Multi-hop reasoning) classified as 83.8% Temporal - should be mixed/factual
2. Category 3 (Temporal reasoning) only 10.4% Temporal - should be ~80%+ Temporal

## Possible Root Causes

### 1. Intent Classification Misfiring

**Evidence**:
- Category 3 (Temporal reasoning) shows only 10.4% Temporal intent
- Category 2 (Multi-hop reasoning) shows 83.8% Temporal intent

**Hypothesis**: Intent classifier is over-triggering on certain patterns

**Check needed**:
```rust
// In query/intent.rs
impl IntentClassifier {
    pub fn classify(&self, query_text: &str) -> ClassifiedIntent {
        // Are temporal patterns too broad?
        // Is there a priority issue?
    }
}
```

### 2. Content-Based Scoring Too Weak

**Evidence**:
- All categories show regression
- Even categories that should benefit (3: Temporal, 4: Entity) regressed

**Hypothesis**: Content-based scores might be too sparse or too weak

**Check needed**:
```rust
// In query/planner.rs
fn temporal_search() -> HashMap<MemoryId, f32> {
    // Are we finding any temporal expressions in queries?
    // Are we matching them to memory expressions?
}

fn entity_search() -> HashMap<MemoryId, f32> {
    // Are we extracting entities correctly?
    // Are we matching them with stop word filtering?
}

fn causal_search() -> HashMap<MemoryId, f32> {
    // Are we detecting causal intent correctly?
    // Are we scoring based on density?
}
```

### 3. Fusion Weights Incorrect

**Evidence**:
- Regression is severe (-29.1% overall)
- Semantic dimension might be getting too little weight

**Current weights** (from query/fusion.rs):
```rust
IntentWeights {
    semantic: 0.50,  // Was: varied by intent
    temporal: 0.35,  // When temporal intent detected
    causal: 0.35,    // When causal intent detected
    entity: 0.35,    // When entity intent detected
}
```

**Hypothesis**: 50% semantic floor might still be too low if other dimensions return sparse scores

### 4. Empty Dimension Scores

**Evidence**:
- Many categories regressed significantly
- Possible that dimensions are returning empty HashMaps

**Hypothesis**: Content-based methods might be returning empty scores, causing fusion to fail

**Check needed**:
```rust
// In query/planner.rs - query() method
let temporal_scores = self.temporal_search(query_text, limit * 2)?;
let entity_scores = self.entity_search(query_text, limit * 2)?;
let causal_scores = self.causal_search(query_text, limit * 2)?;

// Are these returning empty? Log the sizes!
```

### 5. Normalization Issues

**Evidence**:
- Scores might not be normalized correctly across dimensions
- Could cause improper fusion

**Hypothesis**: Content-based scores might have different ranges than semantic scores

## Investigation Plan

### Step 1: Add Debug Logging

Modify `query/planner.rs` to log:
```rust
pub fn query(&self, query_text: &str, ...) -> Result<(ClassifiedIntent, Vec<FusedResult>)> {
    let intent = self.classifier.classify(query_text);

    eprintln!("Query: {}", query_text);
    eprintln!("Intent: {:?}", intent);

    let semantic_scores = self.semantic_search(query_embedding, limit * 2)?;
    eprintln!("Semantic scores: {} results", semantic_scores.len());

    let temporal_scores = self.temporal_search(query_text, limit * 2)?;
    eprintln!("Temporal scores: {} results", temporal_scores.len());

    let entity_scores = self.entity_search(query_text, limit * 2)?;
    eprintln!("Entity scores: {} results", entity_scores.len());

    let causal_scores = self.causal_search(query_text, limit * 2)?;
    eprintln!("Causal scores: {} results", causal_scores.len());

    // ... fusion ...
}
```

### Step 2: Run Debug Query Script

Create a test script to check a few sample queries:
```python
# debug_4d_fusion.py
queries = [
    "What happened yesterday?",  # Should be Temporal
    "Why was the meeting cancelled?",  # Should be Causal
    "Tell me about Alice",  # Should be Entity
    "What is machine learning?",  # Should be Factual
]

for query in queries:
    intent, results = engine.query(query, embedding, limit=5)
    print(f"Query: {query}")
    print(f"Intent: {intent}")
    print(f"Results: {len(results)}")
    for r in results[:3]:
        print(f"  - {r.scores}")
```

### Step 3: Compare with Baseline

Check if `engine.search()` (semantic-only) still works:
```python
# Compare semantic-only vs 4D fusion
semantic_results = engine.search(query_embedding, top_k=10)
fusion_results = engine.query(query_text, query_embedding, limit=10)

print(f"Semantic-only: {len(semantic_results)} results")
print(f"4D fusion: {len(fusion_results[1])} results")
```

### Step 4: Check Content Extraction

Verify that temporal/entity/causal extraction is working:
```rust
#[test]
fn test_content_extraction_on_locomo_data() {
    let temporal_extractor = get_temporal_extractor();
    let entity_extractor = SimpleEntityExtractor::new();
    let causal_extractor = get_causal_extractor();

    // Sample LoCoMo conversation turn
    let text = "I went to the park yesterday with Alice";

    let temporal = temporal_extractor.extract(text);
    println!("Temporal: {:?}", temporal);  // Should find "yesterday"

    let entities = entity_extractor.extract(text).unwrap();
    println!("Entities: {:?}", entities);  // Should find "Alice"

    let (causal, density) = causal_extractor.extract(text);
    println!("Causal: {:?}, Density: {}", causal, density);  // Should be empty
}
```

## Next Steps

1. **STOP** implementing new features
2. **DEBUG** the current implementation:
   - Add logging to query method
   - Run debug queries to understand what's happening
   - Check if dimensions are returning scores
   - Verify intent classification is correct
3. **ANALYZE** the root cause:
   - Is it intent classification?
   - Is it dimension scoring?
   - Is it fusion weights?
   - Is it normalization?
4. **FIX** the identified issue
5. **RE-RUN** benchmark to validate fix

## Questions to Answer

1. **Are content-based methods being called?**
   - Add logging to verify query() is using new methods

2. **Are they returning scores?**
   - Check HashMap sizes for each dimension

3. **Is intent classification working?**
   - Why is Category 3 (Temporal) classified as 74% Factual?
   - Why is Category 2 (Multi-hop) classified as 83.8% Temporal?

4. **Are fusion weights correct?**
   - 50% semantic floor might be too low if other dimensions are sparse

5. **Is normalization working?**
   - Content-based scores might have different ranges

## Conclusion

The content-based dimension improvements (Sprint 16.1-16.3) have caused a **-29.1% regression** in overall performance. This is a critical failure that requires immediate investigation and debugging.

The most likely issues are:
1. Intent classification misfiring (wrong intents for categories)
2. Content-based scoring too sparse (returning empty or very few scores)
3. Fusion weights incorrect (semantic getting too little weight)

**Priority**: HIGH - Stop feature development, debug the regression

---

**Status**: ❌ REGRESSION DETECTED - Investigation Required
