"""
Deeper trace: check entity detection, profile sources, and why
fusion still produces only 20 candidates despite entity exemption.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import mnemefusion
from sentence_transformers import SentenceTransformer

DB_S24 = "tests/benchmarks/fixtures/eval_session24_multipass.mfdb"

def main():
    mem = mnemefusion.Memory(DB_S24, {'embedding_dim': 768})
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    print(f"DB: {mem.count()} memories")

    # Check Melanie profile
    mel_profile = mem.get_entity_profile("melanie")
    mel_short = mem.get_entity_profile("mel")
    if mel_profile:
        n_src = len(mel_profile.get('source_memories', []))
        n_facts = sum(len(v) for v in mel_profile.get('facts', {}).values())
        print(f"Melanie profile: {n_facts} facts, {n_src} sources")
    else:
        print("Melanie profile: NOT FOUND")

    if mel_short:
        n_src = len(mel_short.get('source_memories', []))
        n_facts = sum(len(v) for v in mel_short.get('facts', {}).values())
        print(f"Mel profile: {n_facts} facts, {n_src} sources")
    else:
        print("Mel profile: NOT FOUND")

    # Test entity detection via the query path
    # We can't call detect_entities directly, but we can infer from results
    q = "What does Melanie do to destress?"
    emb = model.encode(q).tolist()

    # Query with different limits
    for lim in [25, 50, 100, 200, 500, 1000]:
        intent_info, results = mem.query(q, emb, limit=lim)
        n = len(results)
        real = sum(1 for m, _ in results if m.get('metadata', {}).get('_source', '') not in ('profile_summary', 'profile_fact'))
        print(f"  limit={lim:4d} -> {n:4d} total ({real:4d} real)")

    # Check: what are the entity scores in the top-20?
    print()
    intent_info, results = mem.query(q, emb, limit=200)
    print(f"Query: {q}")
    print(f"Total results: {len(results)}")

    for i, (mem_dict, scores_dict) in enumerate(results[:25]):
        content = mem_dict.get('content', '')
        meta = mem_dict.get('metadata', {})
        src = meta.get('_source', '')
        sem = scores_dict.get('semantic_score', 0)
        bm25 = scores_dict.get('bm25_score', 0)
        ent = scores_dict.get('entity_score', 0)
        temp = scores_dict.get('temporal_score', 0)
        fused = scores_dict.get('fused_score', 0)

        if src in ('profile_summary', 'profile_fact'):
            print(f"  {i:3d} [PROFILE] ent={ent:.2f} fused={fused:.4f} | {content[:80]}")
        else:
            speaker = meta.get('speaker', '?')
            dia_id = meta.get('dialog_id', '?')
            print(f"  {i:3d} sem={sem:.3f} bm={bm25:.3f} ent={ent:.2f} temp={temp:.3f} fused={fused:.4f} | [{speaker}:{dia_id}] {content[:60]}")

    # Now test a multi-entity query to verify Bug #2 fix
    print()
    print("=" * 80)
    print("MULTI-ENTITY TEST (Bug #2 fix)")
    print("=" * 80)

    multi_q = "How long have Mel and her husband been married?"
    multi_emb = model.encode(multi_q).tolist()
    intent_info, results = mem.query(multi_q, multi_emb, limit=200)
    n = len(results)
    real = sum(1 for m, _ in results if m.get('metadata', {}).get('_source', '') not in ('profile_summary', 'profile_fact'))
    ent_nonzero = sum(1 for _, s in results if s.get('entity_score', 0) > 0)
    print(f"Q: {multi_q}")
    print(f"Results: {n} total, {real} real, {ent_nonzero} with entity>0")

    # Compare with single-entity variant
    single_q = "How long has Melanie been married?"
    single_emb = model.encode(single_q).tolist()
    intent_info, results2 = mem.query(single_q, single_emb, limit=200)
    n2 = len(results2)
    real2 = sum(1 for m, _ in results2 if m.get('metadata', {}).get('_source', '') not in ('profile_summary', 'profile_fact'))
    ent_nonzero2 = sum(1 for _, s in results2 if s.get('entity_score', 0) > 0)
    print(f"Q: {single_q}")
    print(f"Results: {n2} total, {real2} real, {ent_nonzero2} with entity>0")

if __name__ == '__main__':
    main()
