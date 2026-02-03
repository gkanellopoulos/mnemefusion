#!/usr/bin/env python3
"""Quick diagnostic: what does retrieval actually return for failing single-hop questions?"""
import json, os, sys, tempfile, time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "mnemefusion-python"))
sys.path.insert(0, str(project_root))

try:
    from mnemefusion_cuda_wrapper import mnemefusion
except ImportError:
    import mnemefusion

from sentence_transformers import SentenceTransformer

# Load dataset
with open("tests/benchmarks/fixtures/locomo10.json") as f:
    data = json.load(f)
conv = data[0]

# Prepare documents (same logic as benchmark)
documents = []
conv_data = conv.get("conversation", {})
session_idx = 1
while f"session_{session_idx}" in conv_data:
    session_date = conv_data.get(f"session_{session_idx}_date_time", f"session_{session_idx}")
    turns = conv_data.get(f"session_{session_idx}", [])
    for turn_idx, turn in enumerate(turns):
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        if not text:
            continue
        documents.append((text, {"session_date": session_date, "speaker": speaker, "session_idx": str(session_idx)}))
    session_idx += 1

print(f"[INFO] {len(documents)} documents to ingest")

# Test questions (failing single-hop from benchmark)
test_questions = [
    ("What books has Melanie read?", "single-hop"),
    ("Where has Melanie camped?", "single-hop"),
    ("What did Caroline research?", "single-hop - PASSED"),
    ("What does Melanie do to destress?", "single-hop"),
    ("What did Melanie paint recently?", "single-hop"),
    ("What kind of art does Caroline make?", "single-hop"),
    ("What is Caroline's relationship status?", "single-hop"),
    ("Where did Caroline move from 4 years ago?", "single-hop"),
]

# Get ground truth from QA
qa_lookup = {}
for qa in conv.get("qa", []):
    qa_lookup[qa["question"]] = qa.get("answer", "N/A")

# Init embedder
print("[INFO] Loading embedder...")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Ingest WITHOUT LLM extraction (fast batch)
with tempfile.TemporaryDirectory() as tmp:
    db_path = os.path.join(tmp, "debug.mfdb")
    mem = mnemefusion.Memory(db_path, {"embedding_dim": 768})
    mem.reserve_capacity(len(documents))

    print("[INFO] Generating embeddings...")
    contents = [d[0] for d in documents]
    embeddings = embedder.encode(contents, show_progress_bar=True).tolist()

    print("[INFO] Batch ingesting (no LLM extraction)...")
    batch = [{"content": contents[i], "embedding": embeddings[i], "metadata": documents[i][1]} for i in range(len(documents))]
    result = mem.add_batch(batch)
    print(f"[OK] Ingested {result['created_count']} documents")

    # Query each test question
    print("\n" + "=" * 70)
    print("RETRIEVAL DIAGNOSTIC")
    print("=" * 70)

    for question, label in test_questions:
        print(f"\n--- [{label}] {question} ---")
        ground_truth = qa_lookup.get(question, "NOT FOUND IN QA")
        print(f"  GROUND TRUTH: {ground_truth[:200]}")

        q_emb = embedder.encode([question], show_progress_bar=False)[0].tolist()
        intent_info, results = mem.query(question, q_emb, 5)
        print(f"  INTENT: {intent_info}")
        print(f"  RETRIEVED ({len(results)} results):")
        for i, (res_dict, scores) in enumerate(results):
            content = res_dict.get("content", "")[:120]
            meta = res_dict.get("metadata", {})
            date = meta.get("session_date", "")
            speaker = meta.get("speaker", "")
            print(f"    [{i+1}] [{date}] {speaker}: {content}")
            print(f"         scores: {scores}")

        # If 0 results from query(), do raw vector search to check if vector index works
        if len(results) == 0:
            print(f"  RAW VECTOR SEARCH (bypasses fusion+aggregator):")
            raw_results = mem.search(q_emb, 5)
            print(f"    Found {len(raw_results)} results:")
            for i, (res_dict, sim) in enumerate(raw_results):
                content = res_dict.get("content", "")[:120]
                meta = res_dict.get("metadata", {})
                speaker = meta.get("speaker", "")
                print(f"    [{i+1}] sim={sim:.4f} {speaker}: {content}")
