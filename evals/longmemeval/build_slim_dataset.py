#!/usr/bin/env python3
"""Build a slim dataset JSON for run_query_bim.py to avoid MemoryError.

Extracts only the fields needed for querying + judging, drops the heavy
haystack_sessions (~265MB -> ~2MB). Uses ijson for streaming to avoid OOM.

Run this STANDALONE — no mnemefusion, torch, or sentence_transformers imports.

Prerequisites:
    pip install ijson
"""
import json
import os
import ijson

SRC = os.path.join(os.path.dirname(__file__), "fixtures", "longmemeval", "longmemeval_s_cleaned.json")
DST = os.path.join(os.path.dirname(__file__), "fixtures", "longmemeval", "longmemeval_s_slim.json")


def main():
    print(f"Streaming {SRC} with ijson ...")
    slim = []
    with open(SRC, "rb") as f:
        for entry in ijson.items(f, "item"):
            # Extract gold turn contents inline
            gold = []
            for session in entry.get("haystack_sessions", []):
                for turn in session:
                    if turn.get("has_answer"):
                        gold.append(turn["content"].strip())

            slim.append({
                "question_id": entry["question_id"],
                "question_type": entry["question_type"],
                "question": entry["question"],
                "answer": entry["answer"],
                "question_date": entry.get("question_date", ""),
                "_gold_turn_contents": gold,
            })

            if len(slim) % 50 == 0:
                print(f"  Processed {len(slim)} entries...", flush=True)

            # Free the heavy entry immediately
            del entry

    print(f"Writing {DST} ({len(slim)} entries)...")
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False)
    size_mb = os.path.getsize(DST) / 1024 / 1024
    print(f"Done: {len(slim)} entries, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
