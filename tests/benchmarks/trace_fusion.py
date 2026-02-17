"""
Trace why Factual intent queries return only ~20 results while
Entity/Temporal return 200+. Tests candidate flow through fusion.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import mnemefusion
from sentence_transformers import SentenceTransformer

DB_S24 = "tests/benchmarks/fixtures/eval_session24_multipass.mfdb"

def main():
    mem = mnemefusion.Memory(DB_S24, {'embedding_dim': 768})
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    total = mem.count()
    print(f"DB: {total} memories")

    # Test queries spanning different intents
    test_queries = [
        "What does Caroline like?",                      # Expected: Entity
        "What does Melanie do to destress?",             # Expected: Factual
        "When did Melanie sign up for pottery classes?",  # Expected: Temporal
        "What events has Caroline attended recently?",    # Expected: Factual
        "What is Melanie's favorite hobby?",              # Expected: Factual
    ]

    for q in test_queries:
        emb = model.encode(q).tolist()
        for lim in [25, 50, 200]:
            intent_info, results = mem.query(q, emb, limit=lim)
            intent = intent_info.get('intent', '?')
            conf = intent_info.get('confidence', 0)
            n = len(results)
            print(f"  limit={lim:3d} -> {n:3d} results  intent={intent:10s} conf={conf:.2f} | Q: {q[:60]}")
        print()

    # Deep trace: score distribution for a Factual query
    print("=" * 80)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for q in test_queries:
        emb = model.encode(q).tolist()
        intent_info, results = mem.query(q, emb, limit=200)
        intent = intent_info.get('intent', '?')
        n = len(results)

        if not results:
            print(f"\n  Q: {q[:60]}")
            print(f"  Intent: {intent}, Results: {n} (EMPTY)")
            continue

        # Unpack: each result is (mem_dict, scores_dict)
        sem_scores = []
        bm25_scores = []
        ent_scores = []
        fused_scores = []
        temp_scores = []
        profile_count = 0

        for mem_dict, scores_dict in results:
            content = mem_dict.get('content', '')
            metadata = mem_dict.get('metadata', {})

            if metadata.get('_source', '') in ('profile_summary', 'profile_fact'):
                profile_count += 1
                continue

            sem_scores.append(scores_dict.get('semantic_score', 0))
            bm25_scores.append(scores_dict.get('bm25_score', 0))
            ent_scores.append(scores_dict.get('entity_score', 0))
            temp_scores.append(scores_dict.get('temporal_score', 0))
            fused_scores.append(scores_dict.get('fused_score', 0))

        real_count = len(sem_scores)
        print(f"\n  Q: {q[:60]}")
        print(f"  Intent: {intent}, Total: {n}, Real memories: {real_count}, Profile items: {profile_count}")

        if sem_scores:
            print(f"  Semantic:  min={min(sem_scores):.4f} max={max(sem_scores):.4f} mean={sum(sem_scores)/len(sem_scores):.4f}")
            print(f"  BM25:      min={min(bm25_scores):.4f} max={max(bm25_scores):.4f} mean={sum(bm25_scores)/len(bm25_scores):.4f}")
            print(f"  Entity:    min={min(ent_scores):.4f} max={max(ent_scores):.4f} mean={sum(ent_scores)/len(ent_scores):.4f}")
            print(f"  Temporal:  min={min(temp_scores):.4f} max={max(temp_scores):.4f} mean={sum(temp_scores)/len(temp_scores):.4f}")
            print(f"  Fused:     min={min(fused_scores):.4f} max={max(fused_scores):.4f} mean={sum(fused_scores)/len(fused_scores):.4f}")

            # Count entity_score > 0
            ent_nonzero = sum(1 for s in ent_scores if s > 0)
            print(f"  Entity > 0: {ent_nonzero}/{real_count}")

    # Intent routing test
    print()
    print("=" * 80)
    print("INTENT ROUTING TEST")
    print("=" * 80)

    melanie_queries = [
        ("What does Melanie like?", "entity_list_pattern match?"),
        ("What does Melanie do to destress?", "Factual?"),
        ("What hobbies does Melanie have?", "entity_list_pattern?"),
        ("Tell me about Melanie's hobbies", "No question mark"),
        ("What is Melanie interested in?", "entity_list_pattern?"),
        ("How does Melanie relax?", "Factual?"),
    ]

    for q, note in melanie_queries:
        emb = model.encode(q).tolist()
        intent_info, r200 = mem.query(q, emb, limit=200)
        intent = intent_info.get('intent', '?')
        conf = intent_info.get('confidence', 0)
        # Count non-profile results
        real = sum(1 for m, _ in r200 if m.get('metadata', {}).get('_source', '') not in ('profile_summary', 'profile_fact'))
        print(f"  {len(r200):3d} total ({real:3d} real) intent={intent:10s} conf={conf:.2f} | {q:50s} ({note})")

if __name__ == '__main__':
    main()
