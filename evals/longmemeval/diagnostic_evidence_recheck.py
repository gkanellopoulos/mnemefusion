#!/usr/bin/env python3
"""Recheck: Search all 97 NOT_FOUND_R100 cases with lenient exact-substring match."""
import json, os, gc, re, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    import torch
except ImportError:
    pass
from sentence_transformers import SentenceTransformer
import mnemefusion

DB_DIR = "B:/benchmark/bim_phase2_cond0"
ANALYSIS_PATH = os.path.join(DB_DIR, "multihop_entity_analysis.json")
OUTPUT_PATH = os.path.join(DB_DIR, "not_found_recheck.json")


def main():
    with open(ANALYSIS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    not_found = [r for r in data if r["bucket"] == "NOT_FOUND_R100"]
    print(f"Rechecking {len(not_found)} NOT_FOUND_R100 cases with exact substring match")

    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    embed_fn = lambda text: embedder.encode(text, show_progress_bar=False).tolist()
    embedding_dim = embedder.get_sentence_embedding_dimension()

    results_out = []
    found_count = 0
    t_start = time.time()

    for idx, r in enumerate(not_found):
        qid = r["qid"]
        question = r["question"]
        gold = r["gold"]

        db_path = os.path.join(DB_DIR, f"{qid}.mfdb")
        if not os.path.exists(db_path):
            db_path = os.path.join(DB_DIR, f"{qid}_abs.mfdb")
        if not os.path.exists(db_path):
            continue

        config = {"embedding_dim": embedding_dim}
        mem = mnemefusion.Memory(db_path, config)
        mem.set_embedding_fn(embed_fn)
        mem.set_user_entity("user")

        q_emb = embed_fn(question)
        _, query_results, _ = mem.query(question, q_emb, limit=100)

        # Exact substring search (lenient)
        gold_lower = gold.lower().strip()
        # Also try key content words from gold (for multi-word answers)
        gold_words = set(re.findall(r'[a-z]+', gold_lower)) - {
            'the', 'a', 'an', 'of', 'in', 'my', 'i', 'and', 'or', 'to'
        }

        found_positions = []
        for i, (mem_dict, scores) in enumerate(query_results):
            content = mem_dict.get("content", "").lower()
            # Exact substring
            if gold_lower in content:
                found_positions.append((i + 1, "exact", mem_dict.get("content", "")[:80]))
            # For multi-word gold answers: check if ALL content words appear
            elif len(gold_words) >= 2 and all(w in content for w in gold_words):
                found_positions.append((i + 1, "all_words", mem_dict.get("content", "")[:80]))

        if found_positions:
            best_pos = found_positions[0][0]
            found_count += 1
            if best_pos <= 5:
                bucket = "R1-R5"
            elif best_pos <= 10:
                bucket = "R6-R10"
            elif best_pos <= 20:
                bucket = "R11-R20"
            elif best_pos <= 50:
                bucket = "R21-R50"
            else:
                bucket = "R51-R100"
        else:
            bucket = "STILL_NOT_FOUND"

        result = {
            "qid": qid,
            "question": question,
            "gold": gold,
            "original_category": r["category"],
            "new_bucket": bucket,
            "positions": [(p, t, s[:60]) for p, t, s in found_positions[:5]],
        }
        results_out.append(result)

        status = f"{bucket} @{[p[0] for p in found_positions[:3]]}" if found_positions else "STILL_NOT_FOUND"
        print(f"  [{idx+1}/{len(not_found)}] {qid} {status}")

        del mem
        gc.collect()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Found evidence: {found_count}/{len(not_found)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2, ensure_ascii=False)

    # Summary
    from collections import Counter
    buckets = Counter(r["new_bucket"] for r in results_out)
    print(f"\nRevised distribution:")
    for b in ["R1-R5", "R6-R10", "R11-R20", "R21-R50", "R51-R100", "STILL_NOT_FOUND"]:
        count = buckets.get(b, 0)
        pct = count / len(results_out) * 100
        bar = '#' * int(pct / 2)
        print(f"  {b:18s} {count:3d} ({pct:4.1f}%) {bar}")


if __name__ == "__main__":
    main()
