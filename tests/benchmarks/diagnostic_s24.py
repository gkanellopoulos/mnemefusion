"""
S24 Deep Diagnostic: Analyze R@20=0 retrieval failures.

Strategy: Cross-reference the results file (which has retrieved_dialog_ids for ALL 233
questions) with the dataset evidence. For R@20=0 failures, check if the evidence dia_id
appears in ANY other question's retrieved list (proving it exists in the DB). Use entity
profiles from the DB to check profile linkage.

This avoids the DB query limitation where dummy embeddings return few results.
"""

import json
from collections import defaultdict

RESULTS_PATH = "tests/benchmarks/fixtures/session24_2conv_results.json"
DATASET_PATH = "tests/benchmarks/fixtures/locomo10.json"
DB_PATH = "tests/benchmarks/fixtures/eval_session22_3conv.mfdb"

def main():
    # Load data
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        results = json.load(f)
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    questions = results['questions']
    categories = {1: 'single-hop', 2: 'multi-hop', 3: 'temporal', 4: 'open-domain'}

    # Build evidence text map: (sample_id, dia_id) -> {text, speaker}
    evidence_map = {}
    for conv in dataset:
        sample_id = conv.get('sample_id', 'unknown')
        conv_data = conv.get('conversation', {})
        session_idx = 1
        while f'session_{session_idx}' in conv_data:
            turns = conv_data.get(f'session_{session_idx}', [])
            for turn in turns:
                dia_id = turn.get('dia_id', '')
                text = turn.get('text', '')
                speaker = turn.get('speaker', '')
                if dia_id:
                    evidence_map[(sample_id, dia_id)] = {'text': text, 'speaker': speaker}
            session_idx += 1
    print(f"Evidence map: {len(evidence_map)} entries")

    # Build global set of ALL dia_ids that appear in retrieved results across ALL questions.
    # For each question, the retrieved_dialog_ids shows what the system DID retrieve.
    # If a dia_id appears in any question's retrieved list, it exists in the DB.
    all_retrieved_dids = set()
    did_retrieval_details = {}  # dia_id -> list of (question_id, rank) where it appeared
    for q in questions:
        qid = q['question_id']
        for rank, did in enumerate(q.get('retrieved_dialog_ids', [])):
            if did and did.strip():
                all_retrieved_dids.add(did)
                if did not in did_retrieval_details:
                    did_retrieval_details[did] = []
                did_retrieval_details[did].append((qid, rank))

    print(f"Unique dia_ids retrieved across all questions: {len(all_retrieved_dids)}")

    # Also build a map from question to their retrieved_content for analysis
    # The retrieved_content has metadata including entities
    # Format: each entry in retrieved_content is a string like "[2023-05-08] Speaker: text"

    # Open DB for entity profiles only
    import mnemefusion
    mem = mnemefusion.Memory(DB_PATH, {'embedding_dim': 768})
    total_count = mem.count()
    print(f"DB: {total_count} memories")

    profiles = mem.list_entity_profiles()
    profile_map = {}  # lowercase name -> profile
    profile_source_mems = defaultdict(set)  # lowercase name -> set of source memory IDs
    for p in profiles:
        name_lower = p['name'].lower()
        profile_map[name_lower] = p
        for mid in p.get('source_memories', []):
            profile_source_mems[name_lower].add(mid)
        n_facts = sum(len(v) for v in p.get('facts', {}).values())
        n_src = len(p.get('source_memories', []))
        print(f"  Profile: {p['name']}: {n_facts} facts, {n_src} sources")

    # Find R@20=0 failures
    failures = [q for q in questions if q['recall_at_20'] == 0.0 and q['llm_judge_score'] == 0]

    print(f"\n{'='*80}")
    print(f"R@20=0 FAILURES: {len(failures)} / {len(questions)}")
    print(f"{'='*80}")

    # For each failure, check if evidence dia_ids exist in the global retrieved set
    classification = {
        'NOT_IN_DB': [],           # Evidence dia_id never appears in any retrieved list
        'LIKELY_NO_EXTRACTION': [], # In DB (appears in other retrievals) but extraction metadata missing
        'RANKING_PROBLEM': [],     # In DB and retrieved for other queries, just not ranked for this one
        'AMBIGUOUS': [],           # Can't determine status
    }

    # For a more granular classification, let's check the retrieved_content for each
    # question's results. The evidence for a failing question should NOT be in its
    # retrieved_dialog_ids. But if the evidence dia_id appears for OTHER questions,
    # we know it's in the DB.

    for q in failures:
        qid = q['question_id']
        question = q['question']
        category = categories.get(q['category'], 'unknown')
        evidence_ids = q['evidence_ids']
        conv_name = qid.rsplit('_', 1)[0]
        retrieved_dids = set(d for d in q.get('retrieved_dialog_ids', []) if d and d.strip())

        evidence_analysis = []
        for eid in evidence_ids:
            gt = evidence_map.get((conv_name, eid), {})
            gt_text = gt.get('text', '???')
            gt_speaker = gt.get('speaker', '???')

            # Check if this dia_id appears in ANY question's retrieved list
            in_global_retrieved = eid in all_retrieved_dids
            # How often is it retrieved?
            retrieval_count = len(did_retrieval_details.get(eid, []))
            # What's the best rank it achieved?
            best_rank = min((rank for _, rank in did_retrieval_details.get(eid, [(None, 999)])), default=999)

            evidence_analysis.append({
                'eid': eid,
                'gt_text': gt_text[:100],
                'gt_speaker': gt_speaker,
                'in_global_retrieved': in_global_retrieved,
                'retrieval_count': retrieval_count,
                'best_rank': best_rank,
            })

        # Classify based on evidence analysis
        all_in_db = all(ea['in_global_retrieved'] for ea in evidence_analysis)
        none_in_db = all(not ea['in_global_retrieved'] for ea in evidence_analysis)

        if none_in_db:
            classification['NOT_IN_DB'].append((q, evidence_analysis))
        elif all_in_db:
            classification['RANKING_PROBLEM'].append((q, evidence_analysis))
        else:
            classification['AMBIGUOUS'].append((q, evidence_analysis))

    # Print summary
    print(f"\n{'='*80}")
    print(f"FAILURE CLASSIFICATION (based on global retrieval cross-reference)")
    print(f"{'='*80}")
    total_fail = len(failures)
    for cls, items in classification.items():
        pct = len(items)/total_fail*100 if total_fail else 0
        print(f"  {cls:25s}: {len(items):3d} ({pct:.1f}%)")

    # Category breakdown for each class
    for cls, items in classification.items():
        if not items:
            continue
        print(f"\n{'='*80}")
        print(f"  {cls}: {len(items)} failures")
        print(f"{'='*80}")

        cat_counts = {}
        for q, _ in items:
            cat = categories.get(q['category'], 'unknown')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"  By category: {cat_counts}")

        for q, eas in items[:5]:
            qid = q['question_id']
            question = q['question']
            cat = categories.get(q['category'], 'unknown')
            print(f"\n  [{qid}] ({cat})")
            print(f"    Q: {question[:100]}")
            gt = q.get('ground_truth', '???')
            gt_str = str(gt) if not isinstance(gt, str) else gt
            print(f"    A: {gt_str[:80]}")
            print(f"    Evidence: {q['evidence_ids']}")

            for ea in eas:
                status = "IN_DB" if ea['in_global_retrieved'] else "NOT_IN_DB"
                print(f"    {ea['eid']}: {status} (retrieved {ea['retrieval_count']}x, best rank={ea['best_rank']})")
                print(f"      [{ea['gt_speaker']}] {ea['gt_text']}")

        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")

    # Now let's do a deeper analysis of the RANKING_PROBLEM items
    if classification['RANKING_PROBLEM']:
        print(f"\n{'='*80}")
        print(f"RANKING PROBLEM DEEP DIVE")
        print(f"{'='*80}")
        print(f"\nThese {len(classification['RANKING_PROBLEM'])} questions have evidence that IS in the DB")
        print(f"(proven by appearing in other questions' retrieved lists) but not in top-25 for this query.\n")

        # For ranking problems, check if the evidence is frequently retrieved (strong memory)
        # or rarely retrieved (weak memory)
        strong_count = 0  # Evidence retrieved 5+ times for other queries
        weak_count = 0    # Evidence retrieved <5 times
        for q, eas in classification['RANKING_PROBLEM']:
            max_retrievals = max(ea['retrieval_count'] for ea in eas)
            if max_retrievals >= 5:
                strong_count += 1
            else:
                weak_count += 1
        print(f"  Strong memories (retrieved 5+ times elsewhere): {strong_count}")
        print(f"  Weak memories (retrieved <5 times elsewhere): {weak_count}")

        # Check what speaker/entity the evidence relates to
        entity_mentions = defaultdict(int)
        for q, eas in classification['RANKING_PROBLEM']:
            for ea in eas:
                speaker = ea['gt_speaker'].lower()
                entity_mentions[speaker] += 1
        print(f"\n  Evidence by speaker:")
        for speaker, count in sorted(entity_mentions.items(), key=lambda x: -x[1]):
            print(f"    {speaker}: {count} evidence items")

    # NOT_IN_DB analysis — why would evidence not be retrieved anywhere?
    if classification['NOT_IN_DB']:
        print(f"\n{'='*80}")
        print(f"NOT_IN_DB ANALYSIS")
        print(f"{'='*80}")
        print(f"\n{len(classification['NOT_IN_DB'])} questions have evidence that NEVER appears")
        print(f"in any question's retrieved list. Possible reasons:")
        print(f"  1. Evidence was not ingested (missing from DB)")
        print(f"  2. Evidence exists but has very low scores for ALL queries")
        print(f"  3. Evidence dia_id format mismatch between dataset and ingestion")

        # Check if the evidence dia_ids look like they should be in the DB
        missing_dids = set()
        for q, eas in classification['NOT_IN_DB']:
            for ea in eas:
                if not ea['in_global_retrieved']:
                    missing_dids.add(ea['eid'])
        print(f"\n  Unique missing dia_ids: {len(missing_dids)}")
        # Show sample
        for did in sorted(missing_dids)[:10]:
            print(f"    {did}")

    # Impact estimate
    correct = len([q for q in questions if q['llm_judge_score'] == 1])
    print(f"\n{'='*80}")
    print(f"IMPACT ESTIMATE")
    print(f"{'='*80}")
    print(f"  Current correct: {correct}/{len(questions)} = {100*correct/len(questions):.1f}%")
    print(f"  R@20=0 failures: {total_fail}")
    n_not_in_db = len(classification['NOT_IN_DB'])
    n_ranking = len(classification['RANKING_PROBLEM'])
    n_ambig = len(classification['AMBIGUOUS'])
    print(f"    NOT_IN_DB: {n_not_in_db} (need extraction fixes)")
    print(f"    RANKING: {n_ranking} (need scoring/retrieval improvements)")
    print(f"    AMBIGUOUS: {n_ambig}")
    print(f"  If all NOT_IN_DB fixed: +{100*n_not_in_db/len(questions):.1f}% → {100*(correct+n_not_in_db)/len(questions):.1f}%")
    print(f"  If all failures fixed: +{100*total_fail/len(questions):.1f}% → {100*(correct+total_fail)/len(questions):.1f}%")

    # Additional: Check how many questions with judge=0 have R@20>0
    # These are cases where evidence WAS retrieved but the answer was still wrong
    wrong_but_retrieved = [q for q in questions if q['llm_judge_score'] == 0 and q['recall_at_20'] > 0]
    print(f"\n  Judge=0 but R@20>0 (answer wrong despite evidence retrieved): {len(wrong_but_retrieved)}")
    print(f"  Judge=0 and R@20=0 (evidence not retrieved at all): {total_fail}")
    total_wrong = len([q for q in questions if q['llm_judge_score'] == 0])
    print(f"  Total wrong: {total_wrong}")
    print(f"  → {100*total_fail/total_wrong:.1f}% of errors are RETRIEVAL failures")
    print(f"  → {100*len(wrong_but_retrieved)/total_wrong:.1f}% of errors are GENERATION failures (evidence present, answer wrong)")

    return classification

if __name__ == '__main__':
    main()
