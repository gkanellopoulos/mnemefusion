#!/usr/bin/env python3
"""
Retrieval failure analysis — zero OpenAI cost.

For each question in locomo10.json:
  1. Query the DB (top-20)
  2. Check if evidence dialog_ids appear in retrieved results
  3. Categorize: miss (R@20=0), partial (0 < R@20 < 1), hit (R@20=1)

Output:
  - Per-category R@5, R@10, R@20
  - Miss rate / hit rate per category
  - Sample of worst misses for qualitative analysis
  - Retrieval-success vs MCQ-correctness correlation (if results JSON provided)
"""

import json
import sys
import random
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import mnemefusion
except ImportError:
    print("ERROR: mnemefusion not installed")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = str(project_root / "tests/benchmarks/fixtures/eval_s36_phi4_10conv.mfdb")
DATASET_PATH = str(project_root / "tests/benchmarks/fixtures/locomo10.json")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 20
SAMPLE_MISSES = 15   # qualitative sample size

CAT_NAMES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
    5: "Adversarial",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_questions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    qs = []
    for conv in data:
        conv_id = conv.get("sample_id", "?")
        for qa in conv.get("qa", []):
            q   = qa.get("question", "")
            a   = qa.get("answer", "")
            ev  = qa.get("evidence", [])
            cat = qa.get("category", 0)
            if q:
                qs.append(dict(question=q, answer=a, evidence=ev, category=cat, conv_id=conv_id))
    return qs


def recall_at_k(evidence_set, retrieved_ids, k):
    if not evidence_set:
        return None
    return len(evidence_set & set(retrieved_ids[:k])) / len(evidence_set)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Embedding model
    if not _ST_AVAILABLE:
        print("ERROR: sentence-transformers not installed.")
        sys.exit(1)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Dataset
    print(f"Loading dataset: {DATASET_PATH}")
    questions = load_questions(DATASET_PATH)
    print(f"  {len(questions)} questions across all conversations")

    # Filter: only questions with evidence (skip adversarial no-answer)
    with_evidence    = [q for q in questions if q["evidence"]]
    without_evidence = [q for q in questions if not q["evidence"]]
    print(f"  {len(with_evidence)} have evidence, {len(without_evidence)} have no evidence (adversarial)")

    # Open DB
    print(f"Opening DB: {DB_PATH}")
    mem = mnemefusion.Memory(DB_PATH, {"embedding_dim": 768})

    # ── Per-question retrieval ─────────────────────────────────────────────────
    cat_stats = defaultdict(lambda: {
        "n": 0, "r5": 0.0, "r10": 0.0, "r20": 0.0,
        "miss": 0, "partial": 0, "hit": 0,
    })
    all_results = []

    print(f"\nQuerying {len(with_evidence)} questions...")
    for i, q in enumerate(with_evidence):
        if i % 200 == 0:
            print(f"  {i}/{len(with_evidence)} ...", flush=True)

        emb = embedder.encode([q["question"]], show_progress_bar=False)[0].tolist()

        try:
            _, results, _ = mem.query(q["question"], query_embedding=emb, limit=TOP_K)
            retrieved_ids = [r[0]["metadata"].get("dialog_id", "") for r in results]
        except Exception as e:
            retrieved_ids = []

        ev_set = set(q["evidence"])
        r5  = recall_at_k(ev_set, retrieved_ids, 5)
        r10 = recall_at_k(ev_set, retrieved_ids, 10)
        r20 = recall_at_k(ev_set, retrieved_ids, 20)

        s = cat_stats[q["category"]]
        s["n"]   += 1
        s["r5"]  += r5
        s["r10"] += r10
        s["r20"] += r20

        if   r20 == 0.0: s["miss"]    += 1
        elif r20 == 1.0: s["hit"]     += 1
        else:            s["partial"] += 1

        all_results.append(dict(
            question     = q["question"],
            answer       = q["answer"],
            evidence     = q["evidence"],
            retrieved_ids= retrieved_ids,
            r5=r5, r10=r10, r20=r20,
            category     = q["category"],
            conv_id      = q["conv_id"],
        ))

    mem.close()

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("RETRIEVAL ANALYSIS — eval_s36_phi4_10conv.mfdb")
    print("=" * 75)
    print(f"{'Category':<14} {'N':>5} {'R@5':>7} {'R@10':>7} {'R@20':>7} | {'Miss%':>7} {'Part%':>7} {'Hit%':>7}")
    print("-" * 75)

    tot_n = tot_r5 = tot_r10 = tot_r20 = tot_miss = tot_part = tot_hit = 0

    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        n = s["n"]
        if n == 0:
            continue
        r5  = s["r5"]  / n
        r10 = s["r10"] / n
        r20 = s["r20"] / n
        mp  = s["miss"]    / n
        pp  = s["partial"] / n
        hp  = s["hit"]     / n
        nm  = CAT_NAMES.get(cat, f"Cat{cat}")
        print(f"{nm:<14} {n:>5} {r5:>7.1%} {r10:>7.1%} {r20:>7.1%} | {mp:>7.1%} {pp:>7.1%} {hp:>7.1%}")
        tot_n    += n
        tot_r5   += s["r5"]
        tot_r10  += s["r10"]
        tot_r20  += s["r20"]
        tot_miss += s["miss"]
        tot_part += s["partial"]
        tot_hit  += s["hit"]

    print("-" * 75)
    print(f"{'OVERALL':<14} {tot_n:>5} {tot_r5/tot_n:>7.1%} {tot_r10/tot_n:>7.1%} {tot_r20/tot_n:>7.1%} │ {tot_miss/tot_n:>7.1%} {tot_part/tot_n:>7.1%} {tot_hit/tot_n:>7.1%}")
    print()
    print(f"Complete misses (R@20=0): {tot_miss}/{tot_n} = {tot_miss/tot_n:.1%} of questions")
    print(f"At least partial hit:     {tot_part+tot_hit}/{tot_n} = {(tot_part+tot_hit)/tot_n:.1%}")
    print(f"Perfect retrieval (R@20=1): {tot_hit}/{tot_n} = {tot_hit/tot_n:.1%}")

    # ── Qualitative sample of misses ───────────────────────────────────────────
    misses = [r for r in all_results if r["r20"] == 0.0]
    random.seed(42)
    sample = random.sample(misses, min(SAMPLE_MISSES, len(misses)))

    print(f"\n{'─'*75}")
    print(f"SAMPLE OF COMPLETE MISSES (R@20=0)  — {len(misses)} total")
    print(f"{'─'*75}")
    for r in sorted(sample, key=lambda x: x["category"]):
        cat_name = CAT_NAMES.get(r["category"], f"Cat{r['category']}")
        print(f"\n[{cat_name}] {r['conv_id']}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer']}")
        print(f"  Evidence IDs : {r['evidence']}")
        print(f"  Retrieved IDs: {r['retrieved_ids'][:8]}")

    # ── Multi-evidence breakdown ───────────────────────────────────────────────
    multi_ev = [r for r in all_results if len(r["evidence"]) > 1]
    if multi_ev:
        r20_avg = sum(r["r20"] for r in multi_ev) / len(multi_ev)
        miss_n  = sum(1 for r in multi_ev if r["r20"] == 0)
        print(f"\n{'─'*75}")
        print(f"MULTI-EVIDENCE QUESTIONS: {len(multi_ev)} (avg {sum(len(r['evidence']) for r in multi_ev)/len(multi_ev):.1f} pieces)")
        print(f"  R@20 avg: {r20_avg:.1%}   Complete misses: {miss_n}/{len(multi_ev)} = {miss_n/len(multi_ev):.1%}")

    # ── Evidence depth: where does evidence appear in ranking? ─────────────────
    print(f"\n{'─'*75}")
    print("EVIDENCE RANK DISTRIBUTION (for questions where evidence IS retrieved)")
    rank_buckets = defaultdict(int)
    for r in all_results:
        ev_set = set(r["evidence"])
        for rank, rid in enumerate(r["retrieved_ids"], start=1):
            if rid in ev_set:
                bucket = (rank - 1) // 5  # 0→[1-5], 1→[6-10], 2→[11-15], 3→[16-20]
                rank_buckets[bucket] += 1

    total_found = sum(rank_buckets.values())
    if total_found:
        labels = ["1-5", "6-10", "11-15", "16-20"]
        for b, label in enumerate(labels):
            n = rank_buckets[b]
            bar = "█" * int(30 * n / total_found)
            print(f"  Rank {label:>6}: {n:>5}  {bar} {n/total_found:.1%}")

    # ── Save full results ──────────────────────────────────────────────────────
    out_path = project_root / "tests/benchmarks/fixtures/retrieval_analysis.json"
    with open(out_path, "w") as f:
        json.dump({"summary": dict(tot_r5=tot_r5/tot_n, tot_r10=tot_r10/tot_n,
                                   tot_r20=tot_r20/tot_n, tot_miss=tot_miss,
                                   tot_n=tot_n),
                   "results": all_results}, f)
    print(f"\n[Saved full results to {out_path}]")


if __name__ == "__main__":
    main()
