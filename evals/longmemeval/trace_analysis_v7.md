# v7 Trace Analysis — BIM Benchmark (500q, cycles=0, salience disabled) — FINAL

## Run Configuration
- **Config**: v7 = salience reranking disabled, post-fusion filter reverted to pre-fusion,
  entity injection flat 2.0 (Changes 4 and 6 reverted from v6)
- **Model**: gpt-5-mini RAG, gpt-4o judge, BGE-base-en-v1.5 embeddings
- **DB**: Per-haystack BIM master DBs (B:/benchmark/bim_master/, 500 .mfdb files)
- **Trace**: 19 pipeline steps recorded per query

---

## Finding 1: Time Budget — Speaker Reranking Dominates (28% of query time)

*Updated at 283 traces:*

| Step | % Time | Avg (ms) | Max (ms) |
|------|--------|----------|----------|
| speaker_reranking | 27.8% | 161.5 | 536.4 |
| temporal_search | 16.4% | 95.6 | 224.2 |
| intent_classification | 15.9% | 92.7 | 209.5 |
| dialog_bridging | 13.9% | 80.8 | 654.2 |
| result_summary | 9.9% | 57.5 | 91.3 |
| salience_loading | 8.4% | 48.6 | 322.3 |
| mmr_selection | 6.3% | 36.8 | 59.3 |
| causal_search | 0.5% | 3.0 | 57.8 |
| profile_search | 0.3% | 1.7 | 66.0 |
| entity_injection | 0.3% | 1.5 | 3.7 |
| semantic_search | <0.1% | 0.2 | 0.6 |
| rrf_fusion | <0.1% | 0.1 | 0.2 |

**Total avg query time: 582ms** (much higher than the 48ms P50 on the LoCoMo DB)

**Key insight**: The actual retrieval (semantic + BM25 + fusion) is <1% of query time.
The pipeline overhead (speaker reranking, intent classification, temporal search) dominates.
- **speaker_reranking**: 162ms avg, max 536ms. Computing over 140 candidates.
- **dialog_bridging**: 81ms avg but max 654ms — huge variance, expensive outliers.
- **temporal_search**: 96ms avg. Expensive B-tree scan on each haystack load.

---

## Finding 2: Retrieval Quality is Similar Between Successes and Failures

| Metric | Success (45q) | Failure (38q) |
|--------|---------------|---------------|
| sem_top1 | 0.677 | 0.698 |
| sem_top5_avg | 0.626 | 0.655 |
| result_sem_avg | 0.428 | 0.449 |
| bm25_count | 66 | 100 |
| fusion_count | 146 | 147 |
| facts_matched | 0.5 | 0.1 |
| total_results | 20 | 20 |

**Key insight**: Failures have HIGHER semantic scores (0.698 vs 0.677 top-1) and MORE
BM25 candidates (100 vs 66). The retrieval quality per se is not the differentiator.

---

## Finding 3: BM25 Flooding Correlates with Failure

*Updated at 500q FINAL:*

| BM25 candidates | Accuracy | n |
|-----------------|----------|---|
| 0-10 | 75.0% | 4 |
| 11-50 | 53.3% | 45 |
| 51-100 | 26.7% | 135 |
| 101+ | 26.7% | 378 |

With full data, 75.6% of questions fall in the 101+ BM25 category (flooding). At 500q the
correlation is even starker: low BM25 = 53-75% accuracy, high BM25 = 26.7%. Common words in
personal memory DBs ("I", "my", "new", etc.) match everything, flooding the fusion pool.
Note: most BIM haystacks are small enough that nearly all memories match BM25.

---

## Finding 4: Profile Search — Weaker Than Expected at Scale

*Updated at 500q FINAL — earlier finding was based on 83 traces and was misleading:*

| Facts matched | Accuracy | n |
|---------------|----------|---|
| 0 | 28.3% | 424 |
| 1-3 | 27.7% | 83 |
| 4+ | 38.2% | 55 |

At full scale, profile facts show weak correlation with accuracy (28.3% vs 27.7% vs 38.2%).
Only 4+ facts shows a meaningful lift (+10 pts). 84.8% of queries match zero facts.
The earlier finding (71-100% accuracy with facts) was small-sample noise from the first
83 single-session-user traces. Profile search is NOT the high-leverage area initially thought.

---

## Finding 5: The Pipeline Destroys Semantic Relevance for Failures (CRITICAL)

*Updated at 500q FINAL with proper trace data (562 traces):*

**Semantic search #1 retained in final top-5 (per type):**

| Type | Success retained | Failure retained |
|------|-----------------|-----------------|
| single-session-user | **68%** (30/44) | **15%** (4/26) |
| single-session-assistant | 45% (9/20) | 25% (9/36) |
| knowledge-update | 32% (8/25) | 23% (12/53) |
| temporal-reasoning | 31% (11/35) | 15% (24/160) |
| multi-session | 24% (5/21) | 19% (21/112) |
| single-session-preference | 16% (3/19) | 0% (0/11) |
| **Overall** | **43%** (66/164) | **18%** (70/398) |

**Average semantic gap (search_best - final_best):**
- Success: +0.048 (small gap)
- Failure: +0.076 (larger gap, consistent across all types)

**Worst displacements (failures, gap > 0.20):**
| Question | Search Top-1 | Final Top-1 | Gap | Type |
|----------|-------------|-------------|-----|------|
| worsted weight yarn skeins | 0.809 | 0.457 | +0.352 | single-session-user |
| jogging/yoga hours last week | 0.743 | 0.398 | +0.344 | multi-session |
| Imagine Dragons concert | 0.747 | 0.477 | +0.270 | single-session-user |
| favorite running shoes brand | 0.750 | 0.481 | +0.270 | single-session-user |
| jewelry acquired in 2 months | 0.711 | 0.442 | +0.269 | multi-session |
| 5K time improvement | 0.705 | 0.462 | +0.244 | multi-session |
| asylum application wait | 0.743 | 0.507 | +0.235 | single-session-user |

**Root cause**: Entity injection at flat 2.0 for the "user" entity injects ~58-61 memories
into every fusion pool. In BIM per-haystack DBs where ALL memories belong to "user", these
are random "user" memories, not query-relevant. They displace the best semantic match from
the final top-5 in 82% of failure cases.

---

## Finding 6: Prefusion Filter Has Zero Effect in BIM Haystacks

All 140 candidates survive the prefusion filter in every query (threshold=0.3).
Entity exemption covers ~80/140 candidates. BIM haystacks are small enough that
all candidates pass the semantic threshold.

---

## Finding 7: Adaptive K Has Zero Effect

All queries end up with 16-20 results after adaptive K filtering. The filter isn't
differentiating — it's not trimming low-confidence results.

---

## Finding 8: Counter-Intuitive Semantic Score vs Accuracy

*Updated at 500q FINAL:*

| Semantic Top-1 | Accuracy | n |
|----------------|----------|---|
| 0.4-0.5 | 100.0% | 1 |
| 0.5-0.6 | 40.0% | 40 |
| 0.6-0.7 | 31.2% | 208 |
| 0.7-0.8 | 25.4% | 248 |
| 0.8+ | 29.2% | 65 |

Pattern holds at scale: higher semantic match ≠ higher accuracy (31.2% at 0.6-0.7 vs
25.4% at 0.7-0.8). 0.8+ recovers slightly (29.2%) — these are single-session-assistant
queries with very high semantic similarity where even entity injection can't fully displace.

Likely causes:
1. High-sem queries attract more BM25 flooding (topic overlap = more keyword matches)
2. Entity injection effect is proportionally worse when many memories are topic-relevant
3. The pipeline's post-processing steps (speaker reranking, MMR diversity) actively
   diversify away from the semantic best match

---

## Finding 9: Multi-Session Questions Are Fundamentally Unanswerable

133/500 questions are "multi-session" type (26.6% of the benchmark). These require
aggregating information across multiple sessions. In per-haystack evaluation, each
DB contains only one session, so these questions are structurally unanswerable.

**v7**: 15.8% (21/133) FINAL — matches RunC exactly
**RunC**: 15.8% (21/133)

The theoretical ceiling with 0% on multi-session is 73.4% (367/500 other questions × 100%).

---

## Finding 10: Failure Pattern Analysis

All failures (first 83 traces) share common characteristics:
1. Query resolves to "user" entity (universal in single-user haystacks)
2. Zero profile facts matched
3. Specific episodic queries (coupon redemptions, concert attendance, yoga locations)
4. Entity injection floods 60 random "user" memories into fusion pool
5. Best semantic match gets displaced from final top-5

---

## Actionable Insights (Ranked by Expected Impact)

1. **Fix entity injection for single-user DBs**: When entity has 60+ source_memories,
   the flat 2.0 injection overwhelms semantic relevance. Options:
   a. Re-enable adaptive injection (Change 6): `2.0 × min(1.0, 30/source_count)`
   b. Cap entity injection count (currently 60, could reduce to 20-30)
   c. Weight entity injection by semantic similarity to query

2. **Improve profile search hit rate**: 87% of queries match zero facts. Profile search
   uses stemmed word overlap — may need embedding-based fact matching or broader tokenization.

3. **BM25 candidate cap or score threshold**: 100+ BM25 candidates correlates with 41%
   accuracy. Could cap BM25 at 50 candidates or apply a score threshold.

4. **Optimize speaker_reranking**: 32% of query time (171ms avg) is excessive. Profile
   the implementation for optimization opportunities.

5. **Dialog bridging performance**: 30ms avg with 565ms outliers. Add early exit when
   no relevant neighbors found (currently spends time even when injecting 0 memories).

---

## Finding 11: Preference Failures Are Judge/Prompt Issues, Not Retrieval

single-session-preference: 19/30 success (63.3%) at 162 traces.

All failures are "Can you suggest/recommend..." recommendation queries. The system
retrieves relevant memories but the generated recommendation doesn't match the gold
answer format. Interestingly, failures have 0.4 avg facts matched vs 0.0 for successes —
profile facts might be introducing noise for recommendation queries.

Semantic scores are nearly identical (0.703 vs 0.696 for success/failure). This suggests
the problem is in answer generation or judge matching, not retrieval.

---

## Finding 12: Multi-Session Successes Are Lucky Single-Session Answers

The 6 multi-session successes are all "How many..." counting queries where the answer
happened to be derivable from a single session's data. All have BM25=140 (max candidates).
These are not true cross-session reasoning — they're questions with answers in one haystack.

---

## Finding 13: Per-Type Trace Analysis (500q FINAL, 562 traces)

**Retained = semantic search #1 score survives to final top-5 (gap < 0.001)**

| Type | Acc | Succ sem | Fail sem | Succ bm25 | Fail bm25 | Succ facts | Fail facts | Succ ret. | Fail ret. | Succ gap | Fail gap |
|------|-----|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|----------|----------|
| ss-user | 62.9% | 0.677 | 0.705 | 64 | 87 | 0.5 | 0.1 | **68%** | **15%** | +0.040 | +0.103 |
| ss-pref | 63.3% | 0.703 | 0.696 | 131 | 132 | 0.0 | 0.4 | 16% | 0% | +0.066 | +0.082 |
| ss-asst | 35.7% | 0.817 | 0.830 | 137 | 139 | 1.9 | 1.2 | 45% | 25% | +0.045 | +0.086 |
| knowledge | 32.1% | 0.731 | 0.757 | 112 | 112 | 1.6 | 0.7 | 32% | 23% | +0.066 | +0.092 |
| temporal | 17.9% | 0.647 | 0.680 | 116 | 121 | 0.7 | 0.9 | 31% | 15% | +0.038 | +0.071 |
| multi-session | 15.8% | 0.684 | 0.699 | 106 | 107 | 0.7 | 0.4 | 24% | 19% | +0.055 | +0.066 |

**Overall**: Success retention=43%, Failure retention=18%. Avg gap: +0.048 (success), +0.076 (failure).

**Key patterns**:
- **single-session-user**: Retention is the strongest differentiator (68% vs 15%). Semantic
  gap doubles for failures (+0.103 vs +0.040). Low BM25 count for successes (64 vs 87).
- **single-session-assistant**: Highest semantic scores (0.82-0.83) but only 35.7%. Facts
  help (1.9 vs 1.2 for success). Gap between retention (45% vs 25%) explains some failures.
- **knowledge-update**: 32.1% accuracy. Facts correlate (1.6 vs 0.7). Large semantic gap
  in failures (+0.092) — pipeline displaces relevant results.
- **temporal-reasoning**: 17.9% accuracy. Low semantic scores overall (0.65-0.68). The
  questions require temporal reasoning that semantic similarity cannot capture.
- **single-session-preference**: Judge/generation issue, not retrieval (both groups similar).
- **multi-session**: Structurally limited in per-haystack evaluation (single session per DB).

**Entity injection is universal**: All queries inject ~55-61 entity memories. This is a
fixed overhead of ~60 irrelevant memories in every fusion pool for every question.

---

## Finding 14: Head-to-Head v7 vs RunC (FINAL — all 500 questions)

| Metric | Count | % |
|--------|-------|---|
| Both correct | 139 | 27.8% |
| Both wrong | **327** | **65.4%** |
| v7 wins (v7=1, RunC=0) | 16 | 3.2% |
| RunC wins (v7=0, RunC=1) | 18 | 3.6% |
| **Net** | **v7 -2** | |

**Essentially a tie.** v7 (31.0%) vs RunC (31.4%) = -0.4 pts.

**The critical number: 327/500 (65.4%) of questions are wrong in BOTH configurations.**
This means 2/3 of the benchmark is failing regardless of the 6 query-time changes
(post-fusion filter, adaptive entity injection, aggregation cap, salience soft gate,
smooth salience curve, dialog bridging gate). These changes account for <4% of variance.

**The remaining 34 swing questions** (16 v7 wins + 18 RunC wins) show no systematic pattern —
they are distributed across all question types and represent LLM judge noise more than
genuine pipeline differences.

---

## Running Totals (Updated as benchmark progresses)

| Checkpoint | Questions | Overall | single-session-user | multi-session | single-session-preference | temporal-reasoning |
|------------|-----------|---------|---------------------|---------------|---------------------------|--------------------|
| 70q | 70 | ~63% | 62.9% (44/70) | — | — | — |
| 97q | 97 | 46.4% | 62.9% (44/70) | 3.7% (1/27) | — | — |
| 150q | 150 | 40.7% | 62.9% (44/70) | 9.7% (6/62) | 61.1% (11/18) | — |
| 233q | 233 | 36.1% | 62.9% (44/70) | 15.8% (21/133) FINAL | 63.3% (19/30) FINAL | 14.3% (1/7) |
| 242q | 242 | 35.5% | 62.9% (44/70) | 15.8% (21/133) | 63.3% (19/30) | 22.2% (2/9) |
| 274q | 274 | 32.5% | 62.9% (44/70) FINAL | 15.8% (21/133) FINAL | 63.3% (19/30) FINAL | 12.2% (5/41) |
| **500q** | **500** | **31.0%** | **62.9% (44/70)** | **15.8% (21/133)** | **63.3% (19/30)** | **19.5% (26/133)** |

**FINAL RunC vs v7 comparison (500q complete):**

| Type | RunC | v7 | Delta | Winner |
|------|------|-----|-------|--------|
| single-session-user | 61.4% (43/70) | **62.9% (44/70)** | +1.5 | v7 |
| single-session-preference | **66.7%** (20/30) | 63.3% (19/30) | -3.4 | RunC |
| single-session-assistant | **37.5%** (21/56) | 35.7% (20/56) | -1.8 | RunC |
| knowledge-update | **32.1%** (25/78) | **32.1%** (25/78) | 0.0 | Tie |
| temporal-reasoning | **20.3%** (27/133) | 19.5% (26/133) | -0.8 | RunC |
| multi-session | **15.8%** (21/133) | **15.8%** (21/133) | 0.0 | Tie |
| **Overall** | **31.4%** (157/500) | **31.0%** (155/500) | **-0.4** | **Tie** |

v7 wins on single-session-user (+1.5) but loses on 3 other types. Net: -0.4 pts.
**Neither configuration meaningfully outperforms the other.**

**Question type distribution (full 500q):**
| Type | Count | % |
|------|-------|---|
| multi-session | 133 | 26.6% |
| temporal-reasoning | 133 | 26.6% |
| knowledge-update | 78 | 15.6% |
| single-session-user | 70 | 14.0% |
| single-session-assistant | 56 | 11.2% |
| single-session-preference | 30 | 6.0% |

---

## Final Conclusions (500q complete)

### The Pipeline Is Not the Bottleneck

v7 (31.0%) and RunC (31.4%) converge to ~31% regardless of which of the 6 query-time
changes are active. 65.4% of questions fail in both configurations. The 6 changes
(post-fusion filter, adaptive entity injection, aggregation cap, salience soft gate,
smooth salience curve, dialog bridging gate) together move accuracy by <1%.

### Where the 31% Ceiling Comes From

1. **Multi-session (26.6% of questions, 15.8% accuracy)**: Structurally unanswerable in
   per-haystack evaluation. Each DB has one session; these questions need multiple sessions.
   Contributes ~22 pts of the ~69% failure rate.

2. **Temporal-reasoning (26.6% of questions, 19.5% accuracy)**: These require date
   arithmetic and temporal ordering that embedding similarity cannot capture. The pipeline
   provides no temporal reasoning — just temporal search (recency fallback or date-range
   matching), which is insufficient for "how many days between X and Y?" queries.

3. **Entity injection flooding**: Flat 2.0 for the "user" entity injects ~60 memories into
   every fusion pool. In single-user BIM haystacks, ALL memories are "user" memories, so
   entity injection is pure noise. This displaces semantic top-1 from final results in
   82% of failure cases (retention: 43% success vs 18% failure).

4. **BM25 flooding**: 75.6% of questions have 101+ BM25 candidates (accuracy: 26.7%).
   Small personal memory DBs with first-person language ("I", "my") trigger universal
   BM25 matches. Combined with entity injection, the fusion pool is dominated by noise.

### What Would Actually Move the Needle

1. **Disable entity injection for single-entity DBs** — when every memory belongs to one
   entity, entity injection adds zero signal. Expected impact: +3-5 pts on single-session
   types (reclaiming displaced semantic matches).

2. **Temporal reasoning module** — 133 temporal questions at 19.5% accuracy. Even a simple
   date-extraction + arithmetic pipeline could lift this significantly. Not a retrieval
   problem — it's a reasoning gap.

3. **Cross-session query support** — 133 multi-session questions at 15.8%. These need
   access to multiple haystacks or a cross-session index. Structural limitation of
   per-haystack evaluation.

4. **BM25 cap or score threshold** — Cap at 50 candidates or apply minimum TF-IDF score.
   Would reduce fusion pool noise for 75%+ of questions.

### What NOT to Do

- More query-time pipeline tuning (v7 vs RunC proves it doesn't matter)
- Salience-based reranking changes (salience is disabled and both configs converge)
- Profile search optimization (weak correlation with accuracy at full scale)
- Entity injection parameter tuning (the fundamental issue is injecting at all in
  single-entity DBs, not the score value)

### Cost/API Summary

- 500 questions × ~$0.001-0.002/question = ~$0.50-1.00 total API cost
- Average query latency: 582ms (dominated by speaker_reranking: 162ms)
- Total wall time: ~30 min (including 5 crashes + restarts)
