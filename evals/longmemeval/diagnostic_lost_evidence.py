#!/usr/bin/env python3
"""Diagnostic: Compare retrieval for the 12 lost-evidence questions."""
import json, os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    import torch
except ImportError:
    pass
from sentence_transformers import SentenceTransformer
import mnemefusion

DB_DIR = "B:/benchmark/bim_phase2_cond0"
DATASET_PATH = "fixtures/longmemeval/longmemeval_s_cleaned.json"

LOST_QIDS = [
    "f8c5f88b", "5d3d2817", "726462e0", "ad7109d1", "6b168ec8", "86f00804",
    "gpt4_7bc6cf22", "eac54add", "gpt4_7de946e7", "9ea5eabc", "c7dc5443", "8cf51dda",
]

def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    ds_by_q = {d["question_id"]: d for d in dataset}

    with open(os.path.join(DB_DIR, "query_results_v1.json"), encoding="utf-8") as f:
        v1_by_q = {r["question_id"]: r for r in json.load(f)}
    with open(os.path.join(DB_DIR, "query_results.json"), encoding="utf-8") as f:
        v2_by_q = {r["question_id"]: r for r in json.load(f)}

    print("Loading embedding model...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    embed_fn = lambda text: embedder.encode(text, show_progress_bar=False).tolist()

    for qid in LOST_QIDS:
        entry = ds_by_q.get(qid)
        if not entry:
            print(f"\n=== {qid}: NOT IN DATASET ===")
            continue

        question = entry["question"]
        gold = str(entry["answer"])

        db_path = os.path.join(DB_DIR, f"{qid}.mfdb")
        if not os.path.exists(db_path):
            db_path = os.path.join(DB_DIR, f"{qid}_abs.mfdb")
        if not os.path.exists(db_path):
            print(f"\n=== {qid}: DB NOT FOUND ===")
            continue

        v1r = v1_by_q.get(qid, {})
        v2r = v2_by_q.get(qid, {})

        print(f"\n{'='*80}")
        print(f"QID:  {qid}")
        print(f"Q:    {question[:90]}")
        print(f"Gold: {gold[:90]}")
        print(f"V1: score={v1r.get('score','?')}  R@5={v1r.get('recall_at_5',0):.0%}  R@20={v1r.get('recall_at_20',0):.0%}")
        print(f"V2: score={v2r.get('score','?')}  R@5={v2r.get('recall_at_5',0):.0%}  R@20={v2r.get('recall_at_20',0):.0%}")

        embedding_dim = embedder.get_sentence_embedding_dimension()
        config = {"embedding_dim": embedding_dim}
        mem = mnemefusion.Memory(db_path, config)
        mem.set_embedding_fn(embed_fn)
        mem.set_user_entity("user")

        q_emb = embed_fn(question)
        intent_info, results, facts = mem.query(question, q_emb, limit=30)

        # Check gold words
        gold_words = set(gold.lower().split())

        print(f"\nTop-30 results (current v2 code, limit=30):")
        evidence_found_at = None
        for i, (mem_dict, scores) in enumerate(results[:30]):
            content = mem_dict.get("content", "")[:80]
            sal_raw = mem_dict.get("metadata", {}).get("__mf_salience__", "")
            sal_score = None
            if sal_raw:
                try:
                    sal_data = json.loads(sal_raw)
                    sal_score = sal_data.get("score")
                except:
                    pass

            fused = scores.get("fused_score", 0)
            sem = scores.get("semantic_score", 0)
            ent = scores.get("entity_score", 0)
            bm25 = scores.get("bm25_score", 0)
            sal_disp = scores.get("salience_score")

            content_words = set(content.lower().split())
            overlap = len(gold_words & content_words)
            marker = " <<<< EVIDENCE" if overlap >= 3 else ""
            if marker and evidence_found_at is None:
                evidence_found_at = i + 1

            sal_str = f"sal={sal_disp:.2f}" if sal_disp is not None else f"sal(raw)={sal_score:.2f}" if sal_score else "sal=?"
            print(f"  R{i+1:>2}: fused={fused:.4f} sem={sem:.3f} bm25={bm25:.3f} ent={ent:.1f} {sal_str}  {content}{marker}")

        if evidence_found_at:
            print(f"\n  >> Evidence found at R{evidence_found_at} (current v2 code)")
        else:
            print(f"\n  >> Evidence NOT FOUND in top-30")

        # Memory count
        try:
            all_mems = mem.get_memories_by_salience(999999)
            print(f"  >> Total memories: {len(all_mems)}")
        except:
            pass


if __name__ == "__main__":
    main()
