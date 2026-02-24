"""Phi-4-mini vs Qwen3-4B extraction quality comparison.

Runs entity extraction on a sample of LoCoMo documents with both models
and compares: JSON parse success rate, entity count, fact count, fact quality.

Usage:
    python tests/benchmarks/phi4_extraction_test.py --model-a models/qwen3-4b/Qwen3-4B-Instruct-2507.Q4_K_M.gguf --model-b models/phi-4-mini/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf --num-docs 20
"""

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mnemefusion


def load_sample_documents(dataset_path: str, num_docs: int = 20) -> list[dict]:
    """Load sample documents from LoCoMo dataset.

    LoCoMo format: list of conversations, each with:
      conversation.speaker_a, conversation.speaker_b,
      conversation.session_N (list of {speaker, dia_id, text})
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    conv = data[0]  # First conversation
    conversation_id = conv.get("sample_id", "conv-0")
    conversation = conv.get("conversation", {})

    # Iterate sessions in order
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        session = conversation.get(session_key)
        if session is None:
            break

        for turn in session:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "").strip()
            if not text or len(text) < 20:
                continue
            docs.append({
                "conversation_id": conversation_id,
                "speaker": speaker,
                "text": text,
                "dia_id": turn.get("dia_id", ""),
            })
            if len(docs) >= num_docs:
                return docs

        session_idx += 1

    return docs


def run_extraction(model_path: str, docs: list[dict], label: str) -> dict:
    """Run extraction on docs using the specified model.

    Returns dict with stats and per-doc results.
    """
    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"Path:  {model_path}")
    print(f"Docs:  {len(docs)}")
    print(f"{'='*60}")

    # Create a temp DB for this extraction run
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, f"test_{label}.mfdb")
        mem = mnemefusion.Memory(db_path)

        # Enable extraction with the model
        try:
            mem.enable_llm_entity_extraction(model_path, "balanced")
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            return {"error": str(e)}

        results = []
        total_entities = 0
        total_facts = 0
        json_failures = 0
        extraction_errors = 0
        total_time = 0.0

        for i, doc in enumerate(docs):
            speaker = doc["speaker"]
            text = doc["text"]

            start = time.time()
            try:
                # Use extract_text which runs extraction without adding to DB
                extraction = mem.extract_text(text, speaker=speaker)
                elapsed = time.time() - start
                total_time += elapsed

                if extraction is None:
                    extraction_errors += 1
                    results.append({
                        "doc_idx": i,
                        "speaker": speaker,
                        "text": text[:80],
                        "status": "error",
                        "entities": 0,
                        "facts": 0,
                        "time_s": elapsed,
                    })
                    print(f"  [{i+1:2d}/{len(docs)}] ERROR (no result) - {elapsed:.1f}s - {text[:60]}...")
                    continue

                entities = extraction.get("entities", [])
                facts = extraction.get("entity_facts", [])
                records = extraction.get("records", [])
                relationships = extraction.get("relationships", [])

                n_ent = len(entities)
                n_facts = len(facts)
                total_entities += n_ent
                total_facts += n_facts

                results.append({
                    "doc_idx": i,
                    "speaker": speaker,
                    "text": text[:80],
                    "status": "ok",
                    "entities": n_ent,
                    "entity_names": [e.get("name", "?") for e in entities],
                    "facts": n_facts,
                    "fact_details": [
                        f"{f.get('entity','?')}:{f.get('fact_type','?')}={f.get('value','?')}"
                        for f in facts
                    ],
                    "records": len(records),
                    "relationships": len(relationships),
                    "time_s": elapsed,
                })
                print(f"  [{i+1:2d}/{len(docs)}] {n_ent} ent, {n_facts} facts - {elapsed:.1f}s - {text[:60]}...")

            except Exception as e:
                elapsed = time.time() - start
                total_time += elapsed
                err_str = str(e)
                if "JSON" in err_str or "parsing" in err_str.lower():
                    json_failures += 1
                else:
                    extraction_errors += 1
                results.append({
                    "doc_idx": i,
                    "speaker": speaker,
                    "text": text[:80],
                    "status": "json_fail" if "JSON" in err_str else "error",
                    "error": err_str[:200],
                    "time_s": elapsed,
                })
                print(f"  [{i+1:2d}/{len(docs)}] FAIL - {elapsed:.1f}s - {err_str[:80]}")

        # Summary
        n_ok = sum(1 for r in results if r["status"] == "ok")
        avg_time = total_time / len(docs) if docs else 0

        summary = {
            "label": label,
            "model_path": model_path,
            "total_docs": len(docs),
            "successful": n_ok,
            "json_failures": json_failures,
            "extraction_errors": extraction_errors,
            "success_rate": n_ok / len(docs) * 100 if docs else 0,
            "total_entities": total_entities,
            "total_facts": total_facts,
            "avg_entities_per_doc": total_entities / n_ok if n_ok else 0,
            "avg_facts_per_doc": total_facts / n_ok if n_ok else 0,
            "avg_time_s": avg_time,
            "total_time_s": total_time,
            "results": results,
        }

        print(f"\n--- {label} Summary ---")
        print(f"  Success:   {n_ok}/{len(docs)} ({summary['success_rate']:.1f}%)")
        print(f"  JSON fail: {json_failures}")
        print(f"  Errors:    {extraction_errors}")
        print(f"  Entities:  {total_entities} total ({summary['avg_entities_per_doc']:.1f}/doc)")
        print(f"  Facts:     {total_facts} total ({summary['avg_facts_per_doc']:.1f}/doc)")
        print(f"  Avg time:  {avg_time:.1f}s/doc")
        print(f"  Total:     {total_time:.0f}s")

        return summary


def compare_results(summary_a: dict, summary_b: dict):
    """Print side-by-side comparison of two model runs."""
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Model A':>15} {'Model B':>15} {'Delta':>10}")
    print("-" * 65)

    metrics = [
        ("Success rate", "success_rate", "%"),
        ("JSON failures", "json_failures", ""),
        ("Errors", "extraction_errors", ""),
        ("Total entities", "total_entities", ""),
        ("Total facts", "total_facts", ""),
        ("Avg entities/doc", "avg_entities_per_doc", ""),
        ("Avg facts/doc", "avg_facts_per_doc", ""),
        ("Avg time (s)", "avg_time_s", "s"),
    ]

    for label, key, unit in metrics:
        a = summary_a.get(key, 0)
        b = summary_b.get(key, 0)
        delta = b - a
        sign = "+" if delta > 0 else ""
        print(f"  {label:<23} {a:>14.1f} {b:>14.1f} {sign}{delta:>9.1f}{unit}")

    # Per-doc comparison for successful extractions
    print(f"\n--- Per-Document Entity/Fact Comparison ---")
    results_a = {r["doc_idx"]: r for r in summary_a.get("results", []) if r["status"] == "ok"}
    results_b = {r["doc_idx"]: r for r in summary_b.get("results", []) if r["status"] == "ok"}

    common_docs = sorted(set(results_a.keys()) & set(results_b.keys()))
    if common_docs:
        print(f"  {'Doc':>4} {'Speaker':<12} {'A ent':>5} {'B ent':>5} {'A fact':>6} {'B fact':>6}  Text")
        for idx in common_docs[:15]:
            ra = results_a[idx]
            rb = results_b[idx]
            text = ra.get("text", "")[:40]
            print(f"  {idx:>4} {ra['speaker']:<12} {ra['entities']:>5} {rb['entities']:>5} {ra['facts']:>6} {rb['facts']:>6}  {text}...")

    # Show unique facts from each model for a few docs
    print(f"\n--- Fact Quality Sample (first 5 common docs) ---")
    for idx in common_docs[:5]:
        ra = results_a[idx]
        rb = results_b[idx]
        print(f"\n  Doc {idx}: {ra.get('text', '')[:60]}...")
        print(f"    A facts: {ra.get('fact_details', [])}")
        print(f"    B facts: {rb.get('fact_details', [])}")


def main():
    parser = argparse.ArgumentParser(description="Compare extraction quality between two models")
    parser.add_argument("--model-a", required=True, help="Path to first model (e.g., Qwen3-4B)")
    parser.add_argument("--model-b", required=True, help="Path to second model (e.g., Phi-4-mini)")
    parser.add_argument("--label-a", default="Qwen3-4B", help="Label for model A")
    parser.add_argument("--label-b", default="Phi-4-mini", help="Label for model B")
    parser.add_argument("--num-docs", type=int, default=20, help="Number of documents to test")
    parser.add_argument("--dataset", default="tests/benchmarks/fixtures/locomo10.json",
                       help="Path to LoCoMo dataset")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    # Load sample documents
    print(f"Loading {args.num_docs} documents from {args.dataset}...")
    docs = load_sample_documents(args.dataset, args.num_docs)
    print(f"Loaded {len(docs)} documents")

    if not docs:
        print("ERROR: No documents loaded")
        sys.exit(1)

    # Run Model A
    summary_a = run_extraction(args.model_a, docs, args.label_a)
    if "error" in summary_a:
        print(f"Model A failed: {summary_a['error']}")
        sys.exit(1)

    # Run Model B
    # Note: The Rust engine holds the model in memory. We need a new process
    # to load a different model. For now, we'll try loading sequentially
    # (the first model should be freed when the Memory object is garbage collected).
    import gc
    gc.collect()

    summary_b = run_extraction(args.model_b, docs, args.label_b)
    if "error" in summary_b:
        print(f"Model B failed: {summary_b['error']}")
        sys.exit(1)

    # Compare
    compare_results(summary_a, summary_b)

    # Save results
    if args.output:
        output = {
            "model_a": summary_a,
            "model_b": summary_b,
        }
        # Remove per-doc results from summary for cleaner output
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
