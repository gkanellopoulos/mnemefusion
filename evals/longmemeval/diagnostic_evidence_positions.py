#!/usr/bin/env python3
"""Diagnostic: Where does evidence sit in the ranking for failed questions?

Queries EVIDENCE_NOT_FOUND + ENTITY_FLOODING cases with limit=100
to find actual evidence positions using improved heuristic.
"""
import json, os, sys, re, time, gc
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    import torch
except ImportError:
    pass
from sentence_transformers import SentenceTransformer
import mnemefusion

DB_DIR = "B:/benchmark/bim_phase2_cond0"
FAILURES_PATH = os.path.join(DB_DIR, "all_failures_analysis.json")
V3_PATH = os.path.join(DB_DIR, "query_results_v3.json")
OUTPUT_PATH = os.path.join(DB_DIR, "multihop_entity_analysis.json")

def tokenize(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))

STOPWORDS = {'the','a','an','is','was','were','are','been','be','have','has','had',
             'do','does','did','will','would','could','should','may','might','can',
             'shall','to','of','in','for','on','with','at','by','from','as','into',
             'through','during','before','after','above','below','between','out',
             'off','over','under','again','further','then','once','here','there',
             'when','where','why','how','all','both','each','few','more','most',
             'other','some','such','no','nor','not','only','own','same','so','than',
             'too','very','just','about','up','it','its','i','my','me','we','our',
             'you','your','he','she','they','them','their','this','that','these',
             'those','what','which','who','whom','and','but','or','if','while',
             'because','until','although','since','whether','also','any','many'}

def find_evidence_positions(results, question, gold):
    """Find positions where evidence appears using dual heuristic."""
    gold_words = tokenize(gold) - STOPWORDS
    question_words = tokenize(question) - STOPWORDS

    evidence = []
    for i, (mem_dict, scores) in enumerate(results):
        content = mem_dict.get("content", "")
        content_words = tokenize(content) - STOPWORDS

        gold_overlap = len(gold_words & content_words)
        q_overlap = len(question_words & content_words)

        if len(gold_words) <= 2:
            gold_lower = gold.lower().strip()
            content_lower = content.lower()
            if gold_lower in content_lower and q_overlap >= 2:
                evidence.append((i + 1, content[:100], 'exact_short'))
        elif gold_overlap >= 3:
            evidence.append((i + 1, content[:100], f'gold={gold_overlap}'))
        elif gold_overlap >= 2 and q_overlap >= 3:
            evidence.append((i + 1, content[:100], f'gold={gold_overlap}+q={q_overlap}'))

    return evidence


def main():
    with open(FAILURES_PATH, encoding="utf-8") as f:
        all_failures = json.load(f)

    with open(V3_PATH, encoding="utf-8") as f:
        v3_results = json.load(f)
    v3_by_q = {r["question_id"]: r for r in v3_results}

    target = [f for f in all_failures if f["reason"] in ("EVIDENCE_NOT_FOUND", "ENTITY_FLOODING")]
    print(f"Analyzing {len(target)} cases (EVIDENCE_NOT_FOUND + ENTITY_FLOODING)")

    print("Loading embedding model...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    embed_fn = lambda text: embedder.encode(text, show_progress_bar=False).tolist()
    embedding_dim = embedder.get_sentence_embedding_dimension()

    results_out = []
    t_start = time.time()

    for idx, failure in enumerate(target):
        qid = failure["qid"]
        v3r = v3_by_q.get(qid)
        if not v3r:
            print(f"  [{idx+1}/{len(target)}] {qid}: NOT IN V3 RESULTS")
            continue

        question = v3r["question"]
        gold = str(v3r["gold_answer"])
        category = failure["reason"]

        db_path = os.path.join(DB_DIR, f"{qid}.mfdb")
        if not os.path.exists(db_path):
            db_path = os.path.join(DB_DIR, f"{qid}_abs.mfdb")
        if not os.path.exists(db_path):
            print(f"  [{idx+1}/{len(target)}] {qid}: DB NOT FOUND")
            continue

        config = {"embedding_dim": embedding_dim}
        mem = mnemefusion.Memory(db_path, config)
        mem.set_embedding_fn(embed_fn)
        mem.set_user_entity("user")

        q_emb = embed_fn(question)
        intent_info, query_results, facts = mem.query(question, q_emb, limit=100)

        evidence = find_evidence_positions(query_results, question, gold)
        positions = [e[0] for e in evidence]

        if not positions:
            bucket = "NOT_FOUND_R100"
        else:
            best = min(positions)
            if best <= 5:
                bucket = "R1-R5"
            elif best <= 10:
                bucket = "R6-R10"
            elif best <= 20:
                bucket = "R11-R20"
            elif best <= 50:
                bucket = "R21-R50"
            else:
                bucket = "R51-R100"

        result = {
            "qid": qid,
            "category": category,
            "question": question[:120],
            "gold": gold[:80],
            "bucket": bucket,
            "evidence_positions": positions,
            "evidence_count": len(evidence),
            "evidence_details": [(pos, snippet[:80], mtype) for pos, snippet, mtype in evidence[:8]],
            "total_results": len(query_results),
        }
        results_out.append(result)

        status = f"ev@{positions[:5]}" if positions else "NO_EV"
        print(f"  [{idx+1}/{len(target)}] {qid} [{category[:8]}] {bucket:16s} {status}")

        del mem
        gc.collect()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_PATH}")

    # Summary
    from collections import Counter
    print(f"\n{'='*60}")
    print(f"EVIDENCE POSITION DISTRIBUTION ({len(results_out)} cases)")
    print(f"{'='*60}")

    for cat in ["EVIDENCE_NOT_FOUND", "ENTITY_FLOODING"]:
        subset = [r for r in results_out if r["category"] == cat]
        if not subset:
            continue
        buckets = Counter(r["bucket"] for r in subset)
        print(f"\n  {cat} ({len(subset)} cases):")
        for b in ["R1-R5", "R6-R10", "R11-R20", "R21-R50", "R51-R100", "NOT_FOUND_R100"]:
            count = buckets.get(b, 0)
            pct = count / len(subset) * 100
            bar = '#' * int(pct / 2)
            print(f"    {b:16s} {count:3d} ({pct:4.1f}%) {bar}")

    all_buckets = Counter(r["bucket"] for r in results_out)
    print(f"\n  COMBINED ({len(results_out)} cases):")
    for b in ["R1-R5", "R6-R10", "R11-R20", "R21-R50", "R51-R100", "NOT_FOUND_R100"]:
        count = all_buckets.get(b, 0)
        pct = count / len(results_out) * 100
        bar = '#' * int(pct / 2)
        print(f"    {b:16s} {count:3d} ({pct:4.1f}%) {bar}")

    # Multi-hop: evidence piece counts
    with_ev = [r for r in results_out if r["evidence_count"] > 0]
    print(f"\n  Evidence piece distribution ({len(with_ev)} with evidence, {len(results_out) - len(with_ev)} with none):")
    ev_counts = Counter(r["evidence_count"] for r in with_ev)
    for c in sorted(ev_counts):
        print(f"    {c} piece(s): {ev_counts[c]} questions")

    # Show some NOT_FOUND examples for manual inspection
    not_found = [r for r in results_out if r["bucket"] == "NOT_FOUND_R100"]
    if not_found:
        print(f"\n  Sample NOT_FOUND_R100 cases (first 10):")
        for r in not_found[:10]:
            print(f"    {r['qid']}: Q={r['question'][:70]}")
            print(f"             Gold={r['gold'][:50]}")


if __name__ == "__main__":
    main()
