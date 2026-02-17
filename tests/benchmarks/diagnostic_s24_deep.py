"""
S24 Deep Diagnostic v2: Per-dimension score analysis of R@20=0 retrieval failures.

For each of the 62 failing queries, runs the actual query through the system with
limit=100 to find WHERE the evidence memory actually ranks and WHICH dimensions fail.

Classifies failure modes:
- SEMANTIC_GAP: evidence has low semantic similarity to query
- ENTITY_FLOOD: evidence has entity_score but drowned by 100+ same-entity memories
- BM25_MISS: no keyword overlap between query and evidence
- MULTI_DIM_MISS: multiple dimensions failing simultaneously
"""

import json
import sys
import os
from collections import defaultdict, Counter

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

RESULTS_PATH = "tests/benchmarks/fixtures/session24_2conv_results.json"
DATASET_PATH = "tests/benchmarks/fixtures/locomo10.json"
DB_PATH = "tests/benchmarks/fixtures/eval_session24_multipass.mfdb"

def main():
    import mnemefusion
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')

    print("Loading DB...")
    mem = mnemefusion.Memory(DB_PATH, {'embedding_dim': 768})
    print(f"DB: {mem.count()} memories")

    # Build dialog_id -> memory_id mapping
    print("Building dialog_id -> memory_id mapping...")
    all_ids = mem.list_ids()
    did_to_mid = {}  # dialog_id -> memory_id
    mid_to_did = {}  # memory_id -> dialog_id
    mid_to_meta = {} # memory_id -> metadata
    for mid in all_ids:
        m = mem.get(mid)
        if m:
            meta = m.get('metadata', {})
            did = meta.get('dialog_id', '')
            if did:
                did_to_mid[did] = mid
                mid_to_did[mid] = did
                mid_to_meta[mid] = {
                    'dialog_id': did,
                    'speaker': meta.get('speaker', ''),
                    'content': m.get('content', '')[:100],
                    'entity_names': meta.get('entity_names', ''),
                    'conversation_id': meta.get('conversation_id', ''),
                }
    print(f"Mapped {len(did_to_mid)} dialog_ids to memory_ids")

    # Load results and dataset
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        results = json.load(f)
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    questions = results['questions']
    categories = {1: 'single-hop', 2: 'multi-hop', 3: 'temporal', 4: 'open-domain'}

    # Build evidence text map
    evidence_map = {}
    for conv in dataset:
        sample_id = conv.get('sample_id', 'unknown')
        conv_data = conv.get('conversation', {})
        session_idx = 1
        while f'session_{session_idx}' in conv_data:
            for turn in conv_data[f'session_{session_idx}']:
                dia_id = turn.get('dia_id', '')
                if dia_id:
                    evidence_map[(sample_id, dia_id)] = {
                        'text': turn.get('text', ''),
                        'speaker': turn.get('speaker', ''),
                    }
            session_idx += 1

    # Entity profile stats
    profiles = mem.list_entity_profiles()
    profile_source_counts = {}
    for p in profiles:
        name = p['name'].lower()
        n_src = len(p.get('source_memories', []))
        profile_source_counts[name] = n_src
    print(f"\nEntity profile source counts:")
    for name, count in sorted(profile_source_counts.items(), key=lambda x: -x[1]):
        if count >= 10:
            print(f"  {name}: {count} source memories")

    # Find R@20=0 failures
    failures = [q for q in questions if q['recall_at_20'] == 0.0 and q['llm_judge_score'] == 0]
    print(f"\n{'='*80}")
    print(f"RUNNING PER-DIMENSION ANALYSIS ON {len(failures)} R@20=0 FAILURES")
    print(f"{'='*80}")

    # For each failure, run the query with limit=100 and find evidence rank + scores
    failure_analysis = []

    for i, q in enumerate(failures):
        qid = q['question_id']
        question = q['question']
        category = categories.get(q['category'], 'unknown')
        evidence_ids = q['evidence_ids']
        conv_name = qid.rsplit('_', 1)[0]

        # Generate query embedding
        query_emb = model.encode(question).tolist()

        # Run query with limit=100 to find evidence rank
        intent, query_results = mem.query(question, query_emb, 100)

        # Build result map: dialog_id -> (rank, scores)
        result_map = {}  # dialog_id -> (rank, scores, content)
        for rank, (m, scores) in enumerate(query_results):
            meta = m.get('metadata', {})
            did = meta.get('dialog_id', '')
            if did:
                result_map[did] = (rank, scores, m.get('content', '')[:100])

        # Analyze each evidence piece
        evidence_analysis = []
        for eid in evidence_ids:
            gt = evidence_map.get((conv_name, eid), {})
            gt_text = gt.get('text', '???')
            gt_speaker = gt.get('speaker', '???')

            # Check if evidence appears in query results
            if eid in result_map:
                rank, scores, content = result_map[eid]
                evidence_analysis.append({
                    'eid': eid,
                    'gt_text': gt_text,
                    'gt_speaker': gt_speaker,
                    'found_in_results': True,
                    'rank': rank,
                    'scores': scores,
                })
            else:
                # Evidence not even in top-100 — check if it's in DB at all
                mid = did_to_mid.get(eid)
                evidence_analysis.append({
                    'eid': eid,
                    'gt_text': gt_text,
                    'gt_speaker': gt_speaker,
                    'found_in_results': False,
                    'rank': -1,
                    'scores': None,
                    'memory_id': mid,
                    'in_db': mid is not None,
                })

        # What ARE the top-5 results? (what's beating the evidence)
        top5 = []
        for rank, (m, scores) in enumerate(query_results[:5]):
            meta = m.get('metadata', {})
            top5.append({
                'rank': rank,
                'dialog_id': meta.get('dialog_id', ''),
                'speaker': meta.get('speaker', ''),
                'content': m.get('content', '')[:80],
                'scores': scores,
            })

        failure_analysis.append({
            'qid': qid,
            'question': question,
            'category': category,
            'ground_truth': str(q.get('ground_truth', ''))[:100],
            'intent': intent,
            'evidence': evidence_analysis,
            'top5': top5,
            'total_results': len(query_results),
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(failures)} queries...")

    print(f"\nAll {len(failures)} queries processed.\n")

    # === ANALYSIS ===

    # 1. Where does evidence rank?
    print(f"{'='*80}")
    print(f"EVIDENCE RANK DISTRIBUTION")
    print(f"{'='*80}")
    rank_buckets = {'top20': 0, '21-50': 0, '51-100': 0, 'not_in_100': 0, 'not_in_db': 0}
    all_evidence_items = []
    for fa in failure_analysis:
        for ea in fa['evidence']:
            all_evidence_items.append(ea)
            if not ea.get('in_db', True) and not ea.get('found_in_results', False):
                rank_buckets['not_in_db'] += 1
            elif not ea.get('found_in_results', False):
                rank_buckets['not_in_100'] += 1
            elif ea['rank'] < 20:
                rank_buckets['top20'] += 1
            elif ea['rank'] < 50:
                rank_buckets['21-50'] += 1
            else:
                rank_buckets['51-100'] += 1

    total_ev = len(all_evidence_items)
    for bucket, count in rank_buckets.items():
        pct = 100*count/total_ev if total_ev else 0
        print(f"  {bucket:15s}: {count:3d} ({pct:.1f}%)")

    # 2. Per-dimension score analysis for evidence that IS in results
    print(f"\n{'='*80}")
    print(f"PER-DIMENSION SCORES FOR EVIDENCE (where found in top-100)")
    print(f"{'='*80}")

    found_evidence = [ea for ea in all_evidence_items if ea.get('found_in_results')]
    if found_evidence:
        dims = ['semantic_score', 'bm25_score', 'entity_score', 'temporal_score', 'causal_score']
        for dim in dims:
            vals = [ea['scores'][dim] for ea in found_evidence if ea['scores']]
            if vals:
                avg = sum(vals) / len(vals)
                zeros = sum(1 for v in vals if v == 0.0)
                nonzero = len(vals) - zeros
                print(f"  {dim:20s}: avg={avg:.3f}  zero={zeros}/{len(vals)} ({100*zeros/len(vals):.0f}%)  nonzero={nonzero}")

    # 3. Classify failure modes
    print(f"\n{'='*80}")
    print(f"FAILURE MODE CLASSIFICATION")
    print(f"{'='*80}")

    mode_counts = Counter()
    mode_examples = defaultdict(list)

    for fa in failure_analysis:
        modes = set()
        for ea in fa['evidence']:
            if not ea.get('found_in_results'):
                modes.add('NOT_IN_TOP100')
                continue
            s = ea['scores']
            if s['semantic_score'] == 0.0:
                modes.add('SEMANTIC_ZERO')
            if s['entity_score'] == 0.0:
                modes.add('ENTITY_ZERO')
            if s['bm25_score'] == 0.0:
                modes.add('BM25_ZERO')
            if s['semantic_score'] > 0 and s['entity_score'] > 0:
                modes.add('HAS_SIGNAL_BUT_OUTRANKED')

        # Primary classification
        if 'NOT_IN_TOP100' in modes:
            primary = 'NOT_IN_TOP100'
        elif 'SEMANTIC_ZERO' in modes and 'ENTITY_ZERO' in modes and 'BM25_ZERO' in modes:
            primary = 'ALL_DIMENSIONS_ZERO'
        elif 'SEMANTIC_ZERO' in modes and 'ENTITY_ZERO' in modes:
            primary = 'SEMANTIC+ENTITY_ZERO'
        elif 'SEMANTIC_ZERO' in modes:
            primary = 'SEMANTIC_ZERO'
        elif 'ENTITY_ZERO' in modes:
            primary = 'ENTITY_ZERO'
        elif 'HAS_SIGNAL_BUT_OUTRANKED' in modes:
            primary = 'HAS_SIGNAL_OUTRANKED'
        else:
            primary = 'OTHER'

        mode_counts[primary] += 1
        mode_examples[primary].append(fa)

    for mode, count in mode_counts.most_common():
        pct = 100*count/len(failure_analysis) if failure_analysis else 0
        print(f"  {mode:30s}: {count:3d} ({pct:.1f}%)")

    # 4. Detailed examples for each mode
    for mode in mode_counts.keys():
        examples = mode_examples[mode]
        print(f"\n{'='*80}")
        print(f"  {mode}: {len(examples)} failures")
        print(f"{'='*80}")

        # Show category breakdown
        cat_counts = Counter(fa['category'] for fa in examples)
        print(f"  Categories: {dict(cat_counts)}")

        for fa in examples[:3]:
            print(f"\n  [{fa['qid']}] ({fa['category']})")
            print(f"    Q: {fa['question'][:100]}")
            print(f"    A: {fa['ground_truth'][:80]}")

            for ea in fa['evidence']:
                if ea.get('found_in_results'):
                    s = ea['scores']
                    print(f"    Evidence {ea['eid']} -> rank={ea['rank']}")
                    print(f"      sem={s['semantic_score']:.3f} bm25={s['bm25_score']:.3f} "
                          f"ent={s['entity_score']:.3f} temp={s['temporal_score']:.3f} "
                          f"fused={s['fused_score']:.4f}")
                    print(f"      [{ea['gt_speaker']}] {ea['gt_text'][:100]}")
                else:
                    in_db = ea.get('in_db', '?')
                    print(f"    Evidence {ea['eid']} -> NOT IN TOP-100 (in_db={in_db})")
                    print(f"      [{ea['gt_speaker']}] {ea['gt_text'][:100]}")

            # Show what beat the evidence (top-3)
            print(f"    Top-3 retrieved instead:")
            for t in fa['top5'][:3]:
                s = t['scores']
                print(f"      #{t['rank']} did={t['dialog_id']} sem={s['semantic_score']:.3f} "
                      f"bm25={s['bm25_score']:.3f} ent={s['entity_score']:.3f} "
                      f"fused={s['fused_score']:.4f}")
                print(f"        [{t['speaker']}] {t['content']}")

    # 5. Entity flooding analysis
    print(f"\n{'='*80}")
    print(f"ENTITY FLOODING ANALYSIS")
    print(f"{'='*80}")
    print(f"\nFor queries about entities with many source memories, the flat 2.0")
    print(f"entity score means ALL source memories rank equally. Evidence gets lost.\n")

    # Check: how many of the 62 failures mention entities with 100+ source memories?
    entity_flood_count = 0
    for fa in failure_analysis:
        q_lower = fa['question'].lower()
        for name, count in profile_source_counts.items():
            if name in q_lower and count >= 50:
                entity_flood_count += 1
                break
    print(f"  Queries mentioning entities with 50+ source memories: {entity_flood_count}/{len(failures)}")

    # 6. Semantic similarity between query and evidence (direct cosine)
    print(f"\n{'='*80}")
    print(f"DIRECT COSINE SIMILARITY: QUERY vs EVIDENCE TEXT")
    print(f"{'='*80}")
    print(f"(Embedding query and evidence text directly, ignoring DB retrieval)\n")

    import numpy as np

    cosine_sims = []
    for fa in failure_analysis:
        q_emb = model.encode(fa['question'])
        for ea in fa['evidence']:
            if ea['gt_text'] != '???':
                e_emb = model.encode(ea['gt_text'])
                sim = float(np.dot(q_emb, e_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(e_emb)))
                cosine_sims.append({
                    'qid': fa['qid'],
                    'category': fa['category'],
                    'question': fa['question'][:60],
                    'eid': ea['eid'],
                    'evidence_text': ea['gt_text'][:60],
                    'cosine_sim': sim,
                })

    if cosine_sims:
        sims = [c['cosine_sim'] for c in cosine_sims]
        print(f"  Count: {len(sims)}")
        print(f"  Mean:  {np.mean(sims):.3f}")
        print(f"  Median: {np.median(sims):.3f}")
        print(f"  Min:   {np.min(sims):.3f}")
        print(f"  Max:   {np.max(sims):.3f}")
        print(f"  <0.3 (below prefilter): {sum(1 for s in sims if s < 0.3)}/{len(sims)}")
        print(f"  <0.4: {sum(1 for s in sims if s < 0.4)}/{len(sims)}")
        print(f"  <0.5: {sum(1 for s in sims if s < 0.5)}/{len(sims)}")

        # By category
        print(f"\n  By category:")
        for cat in ['single-hop', 'multi-hop', 'temporal', 'open-domain']:
            cat_sims = [c['cosine_sim'] for c in cosine_sims if c['category'] == cat]
            if cat_sims:
                print(f"    {cat:12s}: mean={np.mean(cat_sims):.3f} below_0.3={sum(1 for s in cat_sims if s < 0.3)}/{len(cat_sims)}")

        # Show lowest similarity pairs (hardest cases)
        print(f"\n  LOWEST SIMILARITY (hardest retrieval):")
        for c in sorted(cosine_sims, key=lambda x: x['cosine_sim'])[:10]:
            print(f"    sim={c['cosine_sim']:.3f} [{c['category']}] Q: {c['question']}")
            print(f"             E: {c['evidence_text']}")

    # 7. Summary and recommendations
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    correct = len([q for q in questions if q['llm_judge_score'] == 1])
    print(f"  Current: {correct}/{len(questions)} = {100*correct/len(questions):.1f}%")
    print(f"  R@20=0 failures: {len(failures)} ({100*len(failures)/len(questions):.1f}%)")
    print(f"  If half of R@20=0 fixed: -> {100*(correct + len(failures)//2)/len(questions):.1f}%")
    print(f"  If all R@20=0 fixed:     -> {100*(correct + len(failures))/len(questions):.1f}%")
    print(f"  Target (Mem0):           66.9%")
    print(f"  Gap to Mem0:             {66.9 - 100*correct/len(questions):.1f} pts")
    print(f"  R@20=0 potential:        {100*len(failures)/len(questions):.1f} pts")

if __name__ == '__main__':
    main()
