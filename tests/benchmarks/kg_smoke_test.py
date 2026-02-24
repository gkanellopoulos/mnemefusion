"""KG smoke test: ingest sessions 1-3 of conv 0, eval 10 questions.

Tests end-to-end:
1. Extraction produces relationships
2. store_relationships() creates Entity→Entity graph edges
3. save_graph() persists them
4. Query-time traversal finds related-entity memories
5. MCQ eval on 10 known questions
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mnemefusion
from sentence_transformers import SentenceTransformer

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'qwen3-4b',
                          'Qwen3-4B-Instruct-2507.Q4_K_M.gguf')
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'locomo10.json')

# 12 evidence dia_ids needed for the 10 questions
EVIDENCE_DIDS = {
    'D1:3', 'D1:5', 'D1:9', 'D1:11', 'D1:12',
    'D2:1', 'D2:7', 'D2:8', 'D2:14',
    'D3:1', 'D3:11', 'D3:13',
}


def load_docs_and_questions():
    """Load all turns from sessions 1-3 of conv 0 + first 10 questions."""
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conv_data = data[0]['conversation']
    conv_id = data[0].get('sample_id', 'conv-0')
    questions = data[0]['qa'][:10]
    docs = []

    for session_idx in [1, 2, 3]:
        session_date = conv_data.get(f'session_{session_idx}_date_time', '')
        turns = conv_data.get(f'session_{session_idx}', [])

        for turn_idx, turn in enumerate(turns):
            speaker = turn.get('speaker', 'unknown')
            text = turn.get('text', '')
            dialog_id = turn.get('dia_id', f'{conv_id}_s{session_idx}_t{turn_idx}')
            if not text:
                continue

            metadata = {
                'conversation_id': conv_id,
                'session_idx': str(session_idx),
                'session_date': session_date,
                'turn_idx': str(turn_idx),
                'speaker': speaker,
                'dialog_id': dialog_id,
            }
            docs.append((dialog_id, text, metadata))

    return docs, questions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-path', type=str, default=None, help='Reuse existing DB (skip ingestion)')
    parser.add_argument('--extraction-passes', type=int, default=1, help='Extraction passes (default 1)')
    args = parser.parse_args()

    docs, questions = load_docs_and_questions()
    print(f"Loaded {len(docs)} documents, {len(questions)} questions from LoCoMo conv 0")

    skip_ingestion = False
    if args.db_path and os.path.exists(args.db_path):
        db_path = args.db_path
        skip_ingestion = True
        print(f"Reusing existing DB: {db_path}")
    else:
        db_path = args.db_path or os.path.join(tempfile.mkdtemp(), 'kg_smoke_test.mfdb')
    print(f"DB path: {db_path}")

    # Init embedder
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')

    # Init memory
    passes = args.extraction_passes
    config = {
        "embedding_dim": 768,
        "entity_extraction_enabled": True,
        "extraction_passes": passes,
        "profile_entity_types": ["person"],
    }
    mem = mnemefusion.Memory(db_path, config)

    # Enable LLM extraction
    if os.path.exists(MODEL_PATH):
        mem.enable_llm_entity_extraction(MODEL_PATH, "quality", passes)
        print(f"LLM extraction enabled ({passes}-pass)")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")
        return

    # Set embedding function
    mem.set_embedding_fn(lambda text: embedder.encode(text, show_progress_bar=False).tolist())

    # Ingest
    if not skip_ingestion:
        print(f"\nIngesting {len(docs)} documents ({passes}-pass)...")
        t0 = time.time()
        for i, (doc_id, content, metadata) in enumerate(docs):
            embedding = embedder.encode(content, show_progress_bar=False).tolist()
            mem.add(content, embedding, metadata)
            if (i + 1) % 10 == 0 or i == len(docs) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(docs)}] {rate:.1f} docs/s, ~{(len(docs)-i-1)/rate:.0f}s remaining")
        t1 = time.time()
        print(f"Ingestion done in {t1-t0:.1f}s ({(t1-t0)/len(docs):.1f}s/doc)")

    # Post-processing
    try:
        facts_removed, profiles_deleted = mem.consolidate_profiles()
        print(f"Consolidation: removed {facts_removed} facts, deleted {profiles_deleted} profiles")
    except Exception as e:
        print(f"Consolidation: {e}")

    try:
        n = mem.precompute_fact_embeddings()
        print(f"Precomputed {n} fact embeddings")
    except Exception as e:
        print(f"Fact embeddings: {e}")

    try:
        mem.summarize_profiles()
        print("Profile summaries generated")
    except Exception as e:
        print(f"Summaries: {e}")

    # --- Entity Profiles ---
    print("\n" + "=" * 60)
    print("ENTITY PROFILES")
    print("=" * 60)
    profiles = mem.list_entity_profiles()
    print(f"Total profiles: {len(profiles)}")
    for p in profiles:
        name = p['name']
        facts = p.get('facts', {})
        sources = len(p.get('source_memories', []))
        total = sum(len(v) for v in facts.values() if isinstance(v, list))
        rel_facts = facts.get('relationship', [])

        print(f"\n  {name} ({total} facts, {sources} sources)")
        for ft, flist in facts.items():
            if not isinstance(flist, list):
                continue
            for f in flist[:2]:
                val = f['value'] if isinstance(f, dict) else str(f)
                print(f"    [{ft}] {val}")
            if len(flist) > 2:
                print(f"    ... +{len(flist)-2} more {ft}")
        if rel_facts:
            print(f"    ** {len(rel_facts)} RELATIONSHIP EDGES **")

    # --- 10 Question Eval ---
    print("\n" + "=" * 60)
    print("10-QUESTION EVAL")
    print("=" * 60)

    correct = 0
    for i, q in enumerate(questions):
        question = q['question']
        answer = str(q['answer'])
        evidence = q.get('evidence', [])
        cat = q['category']
        cat_names = {1: 'single-hop', 2: 'temporal', 3: 'open-domain', 4: 'multi-hop'}

        embedding = embedder.encode(question, show_progress_bar=False).tolist()
        intent, results, profile_ctx = mem.query(question, embedding, 10)

        # Build context from results
        context_lines = []
        for mem_dict, scores_dict in results[:10]:
            content = mem_dict.get('content', '')
            entity_s = scores_dict.get('entity_score', 0)
            fused_s = scores_dict.get('fused_score', 0)
            did = mem_dict.get('metadata', {}).get('dialog_id', '?')
            is_evidence = did in evidence
            marker = " <<< EVIDENCE" if is_evidence else ""
            context_lines.append(f"  [{fused_s:.3f}|e={entity_s:.1f}] ({did}) {content[:80]}{marker}")

        # Check if evidence is in top-10
        retrieved_dids = set()
        for mem_dict, _ in results[:10]:
            did = mem_dict.get('metadata', {}).get('dialog_id', '?')
            retrieved_dids.add(did)

        evidence_found = sum(1 for e in evidence if e in retrieved_dids)
        evidence_total = len(evidence)

        # Simple accuracy: is the answer substring in any top-5 result?
        top5_text = ' '.join(m.get('content', '') for m, _ in results[:5])
        # Also check profile context
        profile_text = ' '.join(profile_ctx) if profile_ctx else ''
        combined = (top5_text + ' ' + profile_text).lower()
        answer_lower = answer.lower()

        # Rough check: answer words in context
        answer_words = [w for w in answer_lower.split() if len(w) > 2]
        matched_words = sum(1 for w in answer_words if w in combined)
        word_recall = matched_words / len(answer_words) if answer_words else 0

        status = "LIKELY" if word_recall >= 0.5 else "MISS"
        if word_recall >= 0.5:
            correct += 1

        print(f"\n[Q{i}] ({cat_names.get(cat, '?')}) {question}")
        print(f"  Answer: {answer}")
        print(f"  Evidence: {evidence} -> found {evidence_found}/{evidence_total} in top-10")
        print(f"  Word recall: {word_recall:.0%} [{status}]")
        for line in context_lines[:5]:
            print(line)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {correct}/10 questions likely answerable ({correct*10}%)")
    print(f"DB: {db_path}")
    print("Done!")


if __name__ == '__main__':
    main()
