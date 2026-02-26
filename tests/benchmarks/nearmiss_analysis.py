#!/usr/bin/env python3
"""
Near-miss content analysis — zero OpenAI cost.

For single-hop and temporal complete misses where the correct session IS retrieved
(same Dx: prefix in top-10) but the wrong turn is returned, fetch and compare:
  - Evidence turn content (what we needed)
  - Retrieved near-miss turn content (what we got)

Goal: determine if adjacent turns contain the same info (false negative in evaluation)
or if the ranking function genuinely prefers the wrong turn.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import mnemefusion
except ImportError:
    print("ERROR: mnemefusion not installed")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = str(project_root / "tests/benchmarks/fixtures/eval_s36_phi4_10conv.mfdb")
RESULTS_PATH = str(project_root / "tests/benchmarks/fixtures/retrieval_analysis.json")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 10  # check top-10 for near-misses

CAT_NAMES = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop", 4: "Open-domain", 5: "Adversarial"}

SHOW_SAMPLES = 10   # per category

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_dialog_id(did):
    """'D3:19' -> (3, 19). Returns (None, None) on failure."""
    try:
        parts = did.split(":")
        if len(parts) >= 2 and parts[0].startswith("D"):
            return int(parts[0][1:]), int(parts[1])
    except Exception:
        pass
    return None, None


def same_session(did1, did2):
    s1, _ = parse_dialog_id(did1)
    s2, _ = parse_dialog_id(did2)
    return s1 is not None and s1 == s2


def turn_distance(did1, did2):
    """Returns abs turn difference, or None if unparseable."""
    _, t1 = parse_dialog_id(did1)
    _, t2 = parse_dialog_id(did2)
    if t1 is None or t2 is None:
        return None
    return abs(t1 - t2)


def is_near_miss(evidence_ids, retrieved_ids, k=10):
    """True if all evidence IDs miss top-k but at least one shares session with a retrieved ID."""
    ev_set = set(evidence_ids)
    top_k = set(retrieved_ids[:k])
    if ev_set & top_k:
        return False  # it's actually a hit
    for eid in evidence_ids:
        for rid in retrieved_ids[:k]:
            if same_session(eid, rid):
                return True
    return False


def find_near_miss_pairs(evidence_ids, retrieved_ids, k=10):
    """
    Returns list of (evidence_id, nearest_retrieved_id, turn_dist)
    for each evidence ID that has a near-miss in top-k.
    """
    pairs = []
    for eid in evidence_ids:
        best = None
        best_dist = None
        for rid in retrieved_ids[:k]:
            if same_session(eid, rid):
                dist = turn_distance(eid, rid)
                if dist is not None and (best_dist is None or dist < best_dist):
                    best = rid
                    best_dist = dist
        if best is not None:
            pairs.append((eid, best, best_dist))
    return pairs


# ── Fetch memory content by dialog_id ────────────────────────────────────────

def build_dialog_id_index(mem):
    """
    Scan all memories and build dialog_id -> content map.
    Uses list_all_memories approach via entity profiles or direct storage.
    """
    # Use a fake query to find memories — we'll do a metadata dump instead.
    # Actually: use the search API with a broad embedding and high limit.
    # Simpler: use the fact that memories store metadata including dialog_id.
    # We'll call query() with an embedding + no text to just get storage scan.
    # The most reliable way: use a Python-side search with many results.
    return {}  # placeholder — we'll use a different approach


def fetch_content_by_ids(mem, embedder, query_text, dialog_ids):
    """
    Retrieve a large result set for the question and find the content
    of specific dialog_ids in it.
    """
    target_set = set(dialog_ids)
    emb = embedder.encode([query_text], show_progress_bar=False)[0].tolist()
    try:
        _, results, _ = mem.query(query_text, query_embedding=emb, limit=200)
        found = {}
        for r in results:
            m = r[0]
            did = m.get("metadata", {}).get("dialog_id", "")
            if did in target_set:
                found[did] = {
                    "content": m.get("content", ""),
                    "metadata": m.get("metadata", {}),
                }
        return found
    except Exception as e:
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading retrieval results: {RESULTS_PATH}")
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    results = data["results"]
    print(f"  {len(results)} questions loaded")

    # Find near-miss complete misses for categories 1 (single-hop) and 2 (temporal)
    near_miss_by_cat = defaultdict(list)
    for r in results:
        if r["r20"] > 0:
            continue  # not a complete miss
        cat = r["category"]
        if cat not in (1, 2):
            continue
        pairs = find_near_miss_pairs(r["evidence"], r["retrieved_ids"], k=10)
        if pairs:
            near_miss_by_cat[cat].append({**r, "near_miss_pairs": pairs})

    for cat in (1, 2):
        print(f"\n{CAT_NAMES[cat]}: {len(near_miss_by_cat[cat])} near-miss complete misses")

    # Open DB + embedder to fetch content
    if not _ST_AVAILABLE:
        print("\nWARNING: sentence-transformers not available — skipping content fetch")
        print("Install: pip install sentence-transformers")
        # Still show structural analysis
        for cat in (1, 2):
            show_structural_analysis(cat, near_miss_by_cat[cat])
        return

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Opening DB: {DB_PATH}")
    mem = mnemefusion.Memory(DB_PATH, {"embedding_dim": 768})

    for cat in (1, 2):
        print(f"\n{'=' * 80}")
        print(f"NEAR-MISS DEEP DIVE: {CAT_NAMES[cat]} ({len(near_miss_by_cat[cat])} cases)")
        print(f"{'=' * 80}")

        near_misses = near_miss_by_cat[cat]
        # Sort by turn distance (closest misses first — most interesting)
        near_misses.sort(key=lambda x: min(p[2] for p in x["near_miss_pairs"]))
        sample = near_misses[:SHOW_SAMPLES]

        turn_dist_counts = defaultdict(int)
        false_negative_count = 0

        for nm in near_misses:
            for _, _, dist in nm["near_miss_pairs"]:
                if dist is not None:
                    turn_dist_counts[dist] += 1

        print(f"\nTurn distance distribution (evidence vs retrieved near-miss):")
        for dist in sorted(turn_dist_counts.keys()):
            n = turn_dist_counts[dist]
            bar = "█" * min(40, int(40 * n / max(turn_dist_counts.values())))
            print(f"  Distance={dist:>2}: {n:>4} {bar}")

        print(f"\n{'─' * 80}")
        print(f"SAMPLE {min(SHOW_SAMPLES, len(sample))} NEAREST MISSES (sorted by turn distance)")
        print(f"{'─' * 80}")

        for i, nm in enumerate(sample):
            print(f"\n[{i+1}/{len(sample)}] Conv={nm['conv_id']} | Q: {nm['question']}")
            print(f"  Answer: {nm['answer']}")

            for ev_id, near_id, dist in nm["near_miss_pairs"]:
                print(f"\n  Evidence turn : {ev_id}  |  Retrieved near-miss: {near_id}  |  Distance={dist}")

                # Fetch content of both turns
                all_ids = [ev_id, near_id]
                content_map = fetch_content_by_ids(mem, embedder, nm["question"], all_ids)

                ev_content   = content_map.get(ev_id, {}).get("content", "<not found in top-200>")
                near_content = content_map.get(near_id, {}).get("content", "<not found in top-200>")

                print(f"\n  EVIDENCE ({ev_id}):")
                print(f"    {ev_content[:300]}")

                print(f"\n  RETRIEVED NEAR-MISS ({near_id}):")
                print(f"    {near_content[:300]}")

                # Heuristic: does near-miss content contain answer keywords?
                answer_words = set(str(nm["answer"]).lower().split())
                near_words   = set(near_content.lower().split())
                ev_words     = set(ev_content.lower().split())
                answer_in_near = len(answer_words & near_words) / max(len(answer_words), 1)
                answer_in_ev   = len(answer_words & ev_words) / max(len(answer_words), 1)

                print(f"\n  Answer keyword overlap — Evidence: {answer_in_ev:.0%} | Near-miss: {answer_in_near:.0%}")
                if answer_in_near >= 0.5:
                    false_negative_count += 1
                    print(f"  → LIKELY FALSE NEGATIVE (near-miss also contains answer)")
                else:
                    print(f"  → TRUE MISS (near-miss does NOT contain answer)")

        fn_rate = false_negative_count / max(len(sample), 1)
        print(f"\n{'─' * 80}")
        print(f"FALSE NEGATIVE ESTIMATE ({CAT_NAMES[cat]}): {false_negative_count}/{len(sample)} = {fn_rate:.0%} of near-miss sample")
        print(f"  (A 'false negative' = retrieved near-miss turn contains the answer)")
        print(f"  If high: evaluation is strict, adjacent turns have redundant content")
        print(f"  If low:  ranking function genuinely prefers wrong turn → fixable bug")

    mem.close()
    print("\nDone.")


def show_structural_analysis(cat, near_misses):
    """Structural analysis without DB content fetch."""
    print(f"\n{CAT_NAMES[cat]} near-miss structural analysis:")
    turn_dist_counts = defaultdict(int)
    for nm in near_misses:
        for _, _, dist in nm["near_miss_pairs"]:
            if dist is not None:
                turn_dist_counts[dist] += 1
    for dist in sorted(turn_dist_counts.keys()):
        print(f"  Distance={dist}: {turn_dist_counts[dist]}")


if __name__ == "__main__":
    main()
