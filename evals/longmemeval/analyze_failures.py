#!/usr/bin/env python3
"""
LongMemEval Failure Analysis

Analyzes the 314 failed s-mode questions to identify failure patterns and
determine whether library fixes are warranted.

Analyses:
1. Oracle vs s-mode delta table (per-category)
2. Failure classification: retrieval failure (R@20=0) vs ranking (R@20>0, score=0)
3. Single-session-assistant deep dive (100% oracle → 21.4% s-mode)
4. Multi-session analysis: recall distribution and num_gold_turns correlation
5. Haystack size effect: accuracy vs num_turns
6. Single-session-preference anomaly (83.3% oracle → 90.0% s-mode)
7. Actionability: library-fixable / extraction-fixable / benchmark-specific

Data files (all local):
- longmemeval_results_s_combined.json  (500 s-mode results)
- longmemeval_results_oracle_binary.json  (500 oracle results)
- longmemeval_s_cleaned.json  (raw dataset with gold turn content)

Usage:
    python evals/longmemeval/analyze_failures.py
    python evals/longmemeval/analyze_failures.py --embedding-analysis  # slow, computes cosine sim
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "longmemeval"


def load_results(filename: str) -> list:
    path = FIXTURES_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset() -> dict:
    """Load s_cleaned dataset, return dict keyed by question_id."""
    path = FIXTURES_DIR / "longmemeval_s_cleaned.json"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["question_id"]: item for item in data}


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def analysis_1_delta_table(s_results: list, oracle_results: list):
    """Oracle vs s-mode delta table per category."""
    print_header("1. Oracle vs S-Mode Delta Table")

    oracle_by_id = {r["question_id"]: r for r in oracle_results}
    s_by_id = {r["question_id"]: r for r in s_results}

    # Per-category stats
    categories = sorted(set(r["question_type"] for r in s_results))

    oracle_cat = defaultdict(list)
    s_cat = defaultdict(list)
    for r in oracle_results:
        oracle_cat[r["question_type"]].append(r["score"])
    for r in s_results:
        s_cat[r["question_type"]].append(r["score"])

    print(f"\n  {'Category':<30} {'Oracle':>7} {'S-Mode':>7} {'Delta':>7} {'N':>5}")
    print(f"  {'-' * 58}")

    oracle_total, s_total = [], []
    for cat in categories:
        o_scores = oracle_cat.get(cat, [])
        s_scores = s_cat.get(cat, [])
        o_acc = sum(o_scores) / max(1, len(o_scores)) * 100
        s_acc = sum(s_scores) / max(1, len(s_scores)) * 100
        delta = s_acc - o_acc
        print(f"  {cat:<30} {o_acc:>6.1f}% {s_acc:>6.1f}% {delta:>+6.1f}% {len(s_scores):>5}")
        oracle_total.extend(o_scores)
        s_total.extend(s_scores)

    o_overall = sum(oracle_total) / max(1, len(oracle_total)) * 100
    s_overall = sum(s_total) / max(1, len(s_total)) * 100
    print(f"  {'-' * 58}")
    print(f"  {'OVERALL':<30} {o_overall:>6.1f}% {s_overall:>6.1f}% {s_overall - o_overall:>+6.1f}% {len(s_total):>5}")


def analysis_2_failure_classification(s_results: list):
    """Classify failures as retrieval (R@20=0) vs ranking (R@20>0, score=0)."""
    print_header("2. Failure Classification (S-Mode)")

    failures = [r for r in s_results if r["score"] == 0]
    successes = [r for r in s_results if r["score"] == 1]

    retrieval_fail = [r for r in failures if r.get("recall_at_20", 0) == 0]
    ranking_fail = [r for r in failures if r.get("recall_at_20", 0) > 0]

    print(f"\n  Total questions:     {len(s_results)}")
    print(f"  Correct (score=1):   {len(successes)} ({len(successes)/len(s_results)*100:.1f}%)")
    print(f"  Failed (score=0):    {len(failures)} ({len(failures)/len(s_results)*100:.1f}%)")
    print(f"    Retrieval failure:   {len(retrieval_fail)} (R@20=0, evidence not retrieved)")
    print(f"    Ranking/LLM fail:    {len(ranking_fail)} (R@20>0, evidence found but wrong answer)")

    # Per-category failure breakdown
    print(f"\n  {'Category':<30} {'Fail':>5} {'Retr':>5} {'Rank':>5} {'%Retr':>6}")
    print(f"  {'-' * 53}")

    cats = sorted(set(r["question_type"] for r in failures))
    for cat in cats:
        cat_fail = [r for r in failures if r["question_type"] == cat]
        cat_retr = [r for r in cat_fail if r.get("recall_at_20", 0) == 0]
        cat_rank = [r for r in cat_fail if r.get("recall_at_20", 0) > 0]
        pct_retr = len(cat_retr) / max(1, len(cat_fail)) * 100
        print(f"  {cat:<30} {len(cat_fail):>5} {len(cat_retr):>5} {len(cat_rank):>5} {pct_retr:>5.0f}%")


def analysis_3_ssa_deep_dive(s_results: list, oracle_results: list, dataset: dict):
    """Single-session-assistant deep dive."""
    print_header("3. Single-Session-Assistant Deep Dive")

    ssa_s = [r for r in s_results if r["question_type"] == "single-session-assistant"]
    ssa_oracle = [r for r in oracle_results if r["question_type"] == "single-session-assistant"]
    ssa_failures = [r for r in ssa_s if r["score"] == 0]

    o_acc = sum(r["score"] for r in ssa_oracle) / max(1, len(ssa_oracle)) * 100
    s_acc = sum(r["score"] for r in ssa_s) / max(1, len(ssa_s)) * 100

    print(f"\n  Oracle accuracy:  {o_acc:.1f}% (n={len(ssa_oracle)})")
    print(f"  S-mode accuracy:  {s_acc:.1f}% (n={len(ssa_s)})")
    print(f"  Failures:         {len(ssa_failures)}")

    # Classify SSA failures
    retr_fail = [r for r in ssa_failures if r.get("recall_at_20", 0) == 0]
    rank_fail = [r for r in ssa_failures if r.get("recall_at_20", 0) > 0]
    print(f"    R@20=0 (retrieval):  {len(retr_fail)} ({len(retr_fail)/max(1,len(ssa_failures))*100:.0f}%)")
    print(f"    R@20>0 (ranking):    {len(rank_fail)} ({len(rank_fail)/max(1,len(ssa_failures))*100:.0f}%)")

    # Check if gold turns are assistant-role
    print(f"\n  Gold turn role analysis:")
    assistant_gold_count = 0
    user_gold_count = 0
    mixed_count = 0
    missing_count = 0

    for r in ssa_failures:
        qid = r["question_id"]
        if qid not in dataset:
            missing_count += 1
            continue

        entry = dataset[qid]
        answer_session_ids = set(entry.get("answer_session_ids", []))

        # Find gold turns (those with has_answer=True, or in answer sessions)
        roles = set()
        for session in entry["haystack_sessions"]:
            for turn in session:
                if turn.get("has_answer"):
                    roles.add(turn["role"])

        if not roles:
            # Fallback: check turns in answer sessions
            for sess_idx, session in enumerate(entry["haystack_sessions"]):
                sess_id = entry["haystack_session_ids"][sess_idx] if sess_idx < len(entry.get("haystack_session_ids", [])) else None
                if sess_id in answer_session_ids:
                    for turn in session:
                        roles.add(turn["role"])

        if roles == {"assistant"}:
            assistant_gold_count += 1
        elif roles == {"user"}:
            user_gold_count += 1
        elif len(roles) > 1:
            mixed_count += 1
        else:
            missing_count += 1

    print(f"    Assistant-only gold turns: {assistant_gold_count}")
    print(f"    User-only gold turns:      {user_gold_count}")
    print(f"    Mixed roles:               {mixed_count}")
    print(f"    No gold turns found:       {missing_count}")

    if assistant_gold_count > len(ssa_failures) * 0.5:
        print(f"\n  HYPOTHESIS CONFIRMED: Most SSA failures have assistant-spoken gold content.")
        print(f"  User queries don't semantically match assistant-generated text.")
        print(f"  This is a fundamental embedding asymmetry problem — not library-fixable")
        print(f"  without role-aware ingestion or assistant-content summarization.")

    # Show a few example failures
    print(f"\n  Example SSA failures (first 5):")
    for r in ssa_failures[:5]:
        qid = r["question_id"]
        r20 = r.get("recall_at_20", 0)
        print(f"    [{qid}] R@20={r20:.0%}")
        print(f"      Q: {r['question'][:100]}")
        print(f"      Gold: {r['gold_answer'][:100]}")
        print(f"      Got:  {r['hypothesis'][:100]}")
        print()


def analysis_4_multi_session(s_results: list):
    """Multi-session analysis: recall distribution and correlation with num_gold_turns."""
    print_header("4. Multi-Session Analysis")

    ms = [r for r in s_results if r["question_type"] == "multi-session"]
    ms_fail = [r for r in ms if r["score"] == 0]
    ms_pass = [r for r in ms if r["score"] == 1]

    print(f"\n  Total: {len(ms)}, Pass: {len(ms_pass)}, Fail: {len(ms_fail)}")

    # Recall distribution for failures
    r20_zero = [r for r in ms_fail if r.get("recall_at_20", 0) == 0]
    r20_partial = [r for r in ms_fail if 0 < r.get("recall_at_20", 0) < 1]
    r20_full = [r for r in ms_fail if r.get("recall_at_20", 0) >= 1]

    print(f"\n  Multi-session failure recall distribution:")
    print(f"    R@20=0 (no evidence):     {len(r20_zero)}")
    print(f"    0<R@20<1 (partial):       {len(r20_partial)}")
    print(f"    R@20=1 (full evidence):   {len(r20_full)}")

    # Correlation: accuracy vs num_gold_turns
    gold_buckets = defaultdict(lambda: {"pass": 0, "fail": 0})
    for r in ms:
        ngt = r.get("num_gold_turns", 0)
        bucket = f"{ngt}" if ngt <= 5 else "6+"
        if r["score"] == 1:
            gold_buckets[bucket]["pass"] += 1
        else:
            gold_buckets[bucket]["fail"] += 1

    print(f"\n  Accuracy by num_gold_turns:")
    print(f"  {'Gold Turns':>10} {'Pass':>5} {'Fail':>5} {'Acc':>7}")
    print(f"  {'-' * 30}")
    for bucket in sorted(gold_buckets.keys(), key=lambda x: int(x.replace("+", "99"))):
        p = gold_buckets[bucket]["pass"]
        f = gold_buckets[bucket]["fail"]
        acc = p / max(1, p + f) * 100
        print(f"  {bucket:>10} {p:>5} {f:>5} {acc:>6.1f}%")


def analysis_5_haystack_size(s_results: list):
    """Accuracy vs haystack size (num_turns)."""
    print_header("5. Haystack Size Effect")

    # Bucket by num_turns
    buckets = [
        (0, 100, "0-100"),
        (101, 200, "101-200"),
        (201, 300, "201-300"),
        (301, 500, "301-500"),
        (501, 1000, "501-1000"),
        (1001, float("inf"), "1000+"),
    ]

    print(f"\n  {'Turns':<12} {'N':>5} {'Acc':>7} {'R@20':>7} {'Avg Gold':>9}")
    print(f"  {'-' * 42}")

    for lo, hi, label in buckets:
        subset = [r for r in s_results if lo <= r.get("num_turns", 0) <= hi]
        if not subset:
            continue
        acc = sum(r["score"] for r in subset) / len(subset) * 100
        avg_r20 = sum(r.get("recall_at_20", 0) for r in subset) / len(subset) * 100
        avg_gold = sum(r.get("num_gold_turns", 0) for r in subset) / len(subset)
        print(f"  {label:<12} {len(subset):>5} {acc:>6.1f}% {avg_r20:>6.1f}% {avg_gold:>8.1f}")


def analysis_6_ssp_anomaly(s_results: list, oracle_results: list):
    """Single-session-preference: why does s-mode beat oracle?"""
    print_header("6. Single-Session-Preference Anomaly")

    ssp_s = [r for r in s_results if r["question_type"] == "single-session-preference"]
    ssp_o = [r for r in oracle_results if r["question_type"] == "single-session-preference"]

    o_acc = sum(r["score"] for r in ssp_o) / max(1, len(ssp_o)) * 100
    s_acc = sum(r["score"] for r in ssp_s) / max(1, len(ssp_s)) * 100

    print(f"\n  Oracle accuracy:  {o_acc:.1f}% (n={len(ssp_o)})")
    print(f"  S-mode accuracy:  {s_acc:.1f}% (n={len(ssp_s)})")
    print(f"  Delta:            {s_acc - o_acc:+.1f}%")

    # Find questions that PASS in s-mode but FAIL in oracle
    oracle_by_id = {r["question_id"]: r for r in ssp_o}
    gained = []
    lost = []
    for r in ssp_s:
        o = oracle_by_id.get(r["question_id"])
        if not o:
            continue
        if r["score"] == 1 and o["score"] == 0:
            gained.append((r, o))
        elif r["score"] == 0 and o["score"] == 1:
            lost.append((r, o))

    print(f"\n  Questions gained (oracle=0, s=1): {len(gained)}")
    print(f"  Questions lost (oracle=1, s=0):   {len(lost)}")

    if gained:
        print(f"\n  Gained questions (s-mode correct, oracle wrong):")
        for s_r, o_r in gained[:5]:
            print(f"    [{s_r['question_id']}] Q: {s_r['question'][:80]}")
            print(f"      Gold: {s_r['gold_answer'][:80]}")
            print(f"      Oracle hypothesis: {o_r['hypothesis'][:80]}")
            print(f"      S-mode hypothesis: {s_r['hypothesis'][:80]}")
            print(f"      S-mode R@20: {s_r.get('recall_at_20', 0):.0%}")
            print()

    if gained:
        # Check if gained questions have more context (more turns → more embedding matches)
        avg_turns_gained = sum(r.get("num_turns", 0) for r, _ in gained) / len(gained)
        avg_turns_lost = sum(r.get("num_turns", 0) for r, _ in lost) / len(lost) if lost else 0
        print(f"  Avg turns (gained): {avg_turns_gained:.0f}")
        print(f"  Avg turns (lost):   {avg_turns_lost:.0f}")
        print(f"\n  Hypothesis: More context in s-mode provides additional semantic matches")
        print(f"  for preference questions, compensating for noise from irrelevant turns.")


def analysis_7_actionability(s_results: list, oracle_results: list):
    """Classify failures by actionability."""
    print_header("7. Actionability Summary")

    failures = [r for r in s_results if r["score"] == 0]
    oracle_by_id = {r["question_id"]: r for r in oracle_results}

    # Categories of actionability
    library_fixable = []      # R@20>0 in s-mode AND oracle=1 → retrieval works, ranking/RAG issue
    extraction_fixable = []   # R@20=0 in s-mode BUT oracle=1 → evidence exists, extraction missed it
    embedding_issue = []      # R@20=0 in s-mode AND oracle=0 → fundamental embedding mismatch
    benchmark_specific = []   # Abstention questions or edge cases

    for r in failures:
        qid = r["question_id"]
        o = oracle_by_id.get(qid)
        r20 = r.get("recall_at_20", 0)
        is_abs = r.get("is_abstention", False)

        if is_abs:
            benchmark_specific.append(r)
        elif r20 > 0:
            library_fixable.append(r)
        elif o and o["score"] == 1:
            extraction_fixable.append(r)
        elif o and o["score"] == 0:
            embedding_issue.append(r)
        else:
            benchmark_specific.append(r)

    print(f"\n  Total failures: {len(failures)}")
    print(f"\n  Classification:")
    print(f"    Library-fixable (R@20>0, ranking/RAG issue):    {len(library_fixable)}")
    print(f"    Extraction-fixable (R@20=0, oracle passes):    {len(extraction_fixable)}")
    print(f"    Embedding issue (R@20=0, oracle also fails):   {len(embedding_issue)}")
    print(f"    Benchmark-specific (abstention/edge cases):    {len(benchmark_specific)}")

    # Category breakdown for library-fixable
    if library_fixable:
        print(f"\n  Library-fixable by category:")
        cats = defaultdict(int)
        for r in library_fixable:
            cats[r["question_type"]] += 1
        for cat in sorted(cats, key=cats.get, reverse=True):
            print(f"    {cat:<30} {cats[cat]:>3}")

    # Category breakdown for extraction-fixable
    if extraction_fixable:
        print(f"\n  Extraction-fixable by category:")
        cats = defaultdict(int)
        for r in extraction_fixable:
            cats[r["question_type"]] += 1
        for cat in sorted(cats, key=cats.get, reverse=True):
            print(f"    {cat:<30} {cats[cat]:>3}")

    # Ceiling analysis
    total = len(s_results)
    current_acc = sum(r["score"] for r in s_results) / total * 100
    if_library_fixed = (sum(r["score"] for r in s_results) + len(library_fixable)) / total * 100
    if_extraction_fixed = (sum(r["score"] for r in s_results) + len(library_fixable) + len(extraction_fixable)) / total * 100
    theoretical_max = (sum(r["score"] for r in s_results) + len(failures) - len(embedding_issue)) / total * 100

    print(f"\n  Ceiling analysis:")
    print(f"    Current accuracy:                              {current_acc:.1f}%")
    print(f"    If library-fixable resolved:                   {if_library_fixed:.1f}%  (+{if_library_fixed - current_acc:.1f})")
    print(f"    If library + extraction fixed:                 {if_extraction_fixed:.1f}%  (+{if_extraction_fixed - current_acc:.1f})")
    print(f"    Theoretical max (excl. embedding issues):      {theoretical_max:.1f}%")


def analysis_embedding_similarity(s_results: list, dataset: dict):
    """Compute embedding similarity between queries and gold turns for SSA failures."""
    print_header("BONUS: Embedding Similarity Analysis (SSA Failures)")

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  SKIP: sentence-transformers not installed")
        return

    print("  Loading embedding model...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

    ssa_failures = [r for r in s_results
                    if r["question_type"] == "single-session-assistant" and r["score"] == 0]

    print(f"  Computing similarities for {len(ssa_failures)} SSA failures...")

    sims_assistant = []
    sims_user = []

    for r in ssa_failures:
        qid = r["question_id"]
        if qid not in dataset:
            continue

        entry = dataset[qid]
        question = r["question"]

        # Collect gold turns by role
        for session in entry["haystack_sessions"]:
            for turn in session:
                if turn.get("has_answer"):
                    content = turn["content"].strip()
                    if not content:
                        continue

                    # Compute cosine similarity
                    embs = embedder.encode([question, content], show_progress_bar=False)
                    sim = float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])))

                    if turn["role"] == "assistant":
                        sims_assistant.append(sim)
                    else:
                        sims_user.append(sim)

    if sims_assistant:
        print(f"\n  Query-to-gold similarity (assistant turns):")
        print(f"    Mean: {np.mean(sims_assistant):.4f}")
        print(f"    Std:  {np.std(sims_assistant):.4f}")
        print(f"    Min:  {np.min(sims_assistant):.4f}")
        print(f"    Max:  {np.max(sims_assistant):.4f}")
        print(f"    N:    {len(sims_assistant)}")

    if sims_user:
        print(f"\n  Query-to-gold similarity (user turns):")
        print(f"    Mean: {np.mean(sims_user):.4f}")
        print(f"    Std:  {np.std(sims_user):.4f}")
        print(f"    N:    {len(sims_user)}")

    if sims_assistant and sims_user:
        delta = np.mean(sims_user) - np.mean(sims_assistant)
        print(f"\n  Delta (user - assistant): {delta:+.4f}")
        if delta > 0.05:
            print(f"  CONFIRMED: User queries match user turns better than assistant turns.")
            print(f"  Assistant-spoken gold content has lower semantic overlap with user queries.")


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Failure Analysis")
    parser.add_argument("--embedding-analysis", action="store_true",
                        help="Run embedding similarity analysis (slow, requires sentence-transformers)")
    args = parser.parse_args()

    print("LongMemEval Failure Analysis")
    print("=" * 70)

    # Load data
    print("Loading results...")
    s_results = load_results("longmemeval_results_s_combined.json")
    oracle_results = load_results("longmemeval_results_oracle_binary.json")
    dataset = load_dataset()

    print(f"  S-mode results:  {len(s_results)}")
    print(f"  Oracle results:  {len(oracle_results)}")
    print(f"  Dataset entries: {len(dataset)}")

    # Run analyses
    analysis_1_delta_table(s_results, oracle_results)
    analysis_2_failure_classification(s_results)
    analysis_3_ssa_deep_dive(s_results, oracle_results, dataset)
    analysis_4_multi_session(s_results)
    analysis_5_haystack_size(s_results)
    analysis_6_ssp_anomaly(s_results, oracle_results)
    analysis_7_actionability(s_results, oracle_results)

    if args.embedding_analysis:
        analysis_embedding_similarity(s_results, dataset)

    print(f"\n{'=' * 70}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
