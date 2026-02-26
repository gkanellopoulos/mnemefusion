#!/usr/bin/env python3
"""
Score breakdown diagnostic — for specific near-miss cases, show full score
breakdown (semantic, BM25, entity, temporal, fused) for evidence turn vs
retrieved near-miss turn.

Usage:
    python score_breakdown.py

Hard-coded cases from nearmiss_analysis output.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import mnemefusion
except ImportError:
    print("ERROR: mnemefusion not installed")
    sys.exit(1)

from sentence_transformers import SentenceTransformer

DB_PATH = str(project_root / "tests/benchmarks/fixtures/eval_s36_phi4_10conv.mfdb")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Specific near-miss cases to investigate
# Format: (question, answer, evidence_id, near_miss_id)
CASES = [
    # Temporal — dist=1 examples
    ("When did Maria join a gym?",
     "The week before 16 June 2023",
     "D19:1", "D19:2"),
    ("When did John start boot camp with his family?",
     "April.2023",
     "D13:3", "D13:4"),
    ("When did Melanie's friend adopt a child?",
     "2022",
     "D17:3", "D17:4"),
    # Single-hop — dist=1 examples
    ("What are Maria's dogs' names?",
     "Coco, Shadow",
     "D31:4", "D31:3"),
    ("How many dogs has Maria adopted from the dog shelter she volunteers at?",
     "two",
     "D31:2", "D31:3"),
    ("Why did Gina decide to start her own clothing store?",
     "She always loved fashion trends and finding unique pieces and she lost her job so decided it was time to start her own business.",
     "D1:3", "D1:4"),
]


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Opening DB: {DB_PATH}")
    mem = mnemefusion.Memory(DB_PATH, {"embedding_dim": 768})

    for question, answer, ev_id, near_id in CASES:
        print(f"\n{'='*80}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Evidence: {ev_id}  |  Near-miss: {near_id}")
        print(f"{'='*80}")

        emb = embedder.encode([question], show_progress_bar=False)[0].tolist()

        # Query with limit=200 to find both turns
        try:
            intent, results, profile_ctx = mem.query(question, query_embedding=emb, limit=200)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        print(f"Intent: {intent}")
        print(f"Profile ctx length: {len(profile_ctx)} chars")
        print(f"Total results: {len(results)}")

        # Find both evidence and near-miss turns
        ev_result   = None
        near_result = None
        ev_rank     = None
        near_rank   = None

        for rank, r in enumerate(results, start=1):
            mem_dict, fused = r[0], r[1]
            did = mem_dict.get("metadata", {}).get("dialog_id", "")
            if did == ev_id and ev_result is None:
                ev_result = (rank, mem_dict, fused)
            elif did == near_id and near_result is None:
                near_result = (rank, mem_dict, fused)

        # Print evidence turn info
        if ev_result:
            rank, m, f = ev_result
            print(f"\n[EVIDENCE TURN {ev_id}] — Rank #{rank}")
            print(f"  Content: {m.get('content', '')[:200]}")
            print(f"  Fused score : {f.get('fused_score', '?'):.4f}")
            print(f"  Semantic    : {f.get('semantic_score', '?'):.4f}")
            print(f"  BM25        : {f.get('bm25_score', '?'):.4f}")
            print(f"  Entity      : {f.get('entity_score', '?'):.4f}")
            print(f"  Temporal    : {f.get('temporal_score', '?'):.4f}")
            print(f"  Metadata    : {m.get('metadata', {})}")
        else:
            print(f"\n[EVIDENCE TURN {ev_id}] — NOT FOUND in top-200")
            # Try to check if it exists at all with direct list
            print(f"  → Evidence turn is ranked below 200 or not in DB")

        # Print near-miss turn info
        if near_result:
            rank, m, f = near_result
            print(f"\n[NEAR-MISS TURN {near_id}] — Rank #{rank}")
            print(f"  Content: {m.get('content', '')[:200]}")
            print(f"  Fused score : {f.get('fused_score', '?'):.4f}")
            print(f"  Semantic    : {f.get('semantic_score', '?'):.4f}")
            print(f"  BM25        : {f.get('bm25_score', '?'):.4f}")
            print(f"  Entity      : {f.get('entity_score', '?'):.4f}")
            print(f"  Temporal    : {f.get('temporal_score', '?'):.4f}")
            print(f"  Metadata    : {m.get('metadata', {})}")
        else:
            print(f"\n[NEAR-MISS TURN {near_id}] — NOT FOUND in top-200")

        # Top-5 for reference
        print(f"\nTop-5 results:")
        for rank, r in enumerate(results[:5], start=1):
            m, f = r[0], r[1]
            did  = m.get("metadata", {}).get("dialog_id", "")
            fused = f.get("fused_score", 0)
            sem   = f.get("semantic_score", 0)
            bm25  = f.get("bm25_score", 0)
            ent   = f.get("entity_score", 0)
            print(f"  #{rank} [{did:>8}] fused={fused:.3f} sem={sem:.3f} bm25={bm25:.3f} ent={ent:.3f} | {m.get('content','')[:80]}")

    mem.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
