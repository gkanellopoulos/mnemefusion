#!/usr/bin/env python3
"""
Failure Analysis for 885-question benchmark.
Categorizes incorrect answers by root cause:
  - retrieval_failure: evidence NOT in top-20 results (R@20=0)
  - partial_retrieval: SOME evidence in top-20 but not all
  - generation_failure: ALL evidence in top-20 but wrong answer
  - no_evidence: question has no evidence IDs
"""
import mnemefusion, json, sys, os, time
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Setup
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
mem = mnemefusion.Memory('tests/benchmarks/fixtures/eval_session26_3pass.mfdb',
                         {'embedding_dim': 768, 'entity_extraction_enabled': True})
llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

with open('tests/benchmarks/fixtures/locomo10.json') as f:
    data = json.load(f)

cats = {1: 'single-hop', 2: 'multi-hop', 3: 'temporal', 4: 'open-domain', 5: 'adversarial'}

# Prepare questions (same as benchmark)
questions = []
for conv_idx, conv in enumerate(data[:6]):
    for qa in conv.get('qa', []):
        cat = qa.get('category', 0)
        if cat not in cats:
            continue
        evidence = qa.get('evidence', [])
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [str(e) for e in evidence if e]
        answer = qa.get('answer', qa.get('adversarial_answer', ''))
        questions.append((qa['question'], answer, cat, conv.get('sample_id', ''), evidence))

questions = questions[:885]

# Classify failures
failures = {'retrieval': [], 'partial': [], 'generation': [], 'no_evidence': []}
failure_by_cat = {c: {'retrieval': 0, 'partial': 0, 'generation': 0, 'correct': 0, 'no_evidence': 0} for c in cats}
correct_count = 0
total = len(questions)

print(f"Analyzing {total} questions...")
for i, (q, answer, cat, conv_id, evidence) in enumerate(questions):
    if (i+1) % 50 == 0:
        print(f"  [{i+1}/{total}]", flush=True)

    # Query
    emb = model.encode([q], show_progress_bar=False)[0].tolist()
    intent, results, profile_ctx = mem.query(q, emb, 25)

    # Get dialog_ids from results
    retrieved_ids = []
    for result_dict, scores_dict in results:
        metadata = result_dict.get('metadata', {})
        retrieved_ids.append(metadata.get('dialog_id', ''))

    # Build context (same as benchmark)
    contents = []
    for result_dict, scores_dict in results:
        content = result_dict.get('content', '')
        metadata = result_dict.get('metadata', {})
        session_date = metadata.get('session_date', '')
        speaker = metadata.get('speaker', '')
        if session_date:
            formatted = f"[{session_date}] {speaker}: {content}" if speaker else f"[{session_date}] {content}"
        else:
            formatted = f"{speaker}: {content}" if speaker else content
        contents.append(formatted)

    n_profile = min(len(profile_ctx), 5)
    context = profile_ctx[:n_profile] + contents[:25 - n_profile]

    # Generate answer
    context_str = "\n".join([f"- {c}" for c in context[:25]])
    prompt = (
        "You are a helpful assistant answering questions based on conversation history.\n\n"
        "Retrieved memories (dates in brackets show when each conversation occurred):\n"
        f"{context_str}\n\n"
        f"Question: {q}\n\n"
        "Answer the question based on the information in the retrieved memories. "
        "Look carefully through ALL the memories for relevant details.\n"
        "For temporal questions (when did X happen), use the dates in brackets.\n"
        "If you find ANY relevant information, provide an answer. "
        "Only say 'I don\\'t have enough information' if the memories truly contain nothing related.\n"
        "Keep your answer concise and factual."
    )

    try:
        resp = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150, temperature=0
        )
        generated = resp.choices[0].message.content.strip()
    except Exception as e:
        generated = f"Error: {e}"

    # Judge
    judge_prompt = (
        "You are evaluating if an AI assistant's answer is correct compared to a ground truth answer.\n\n"
        f"Question: {q}\n"
        f"Ground Truth Answer: {answer}\n"
        f"AI Answer: {generated}\n\n"
        "Is the AI answer essentially correct? (It doesn't need to be word-for-word identical, "
        "but should convey the same key information)\n\n"
        'Reply with ONLY "yes" or "no".'
    )

    try:
        resp = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=5, temperature=0
        )
        judge = resp.choices[0].message.content.strip().lower()
        is_correct = judge.startswith('yes')
    except:
        is_correct = False

    # Categorize
    evidence_set = set(evidence)
    if is_correct:
        failure_by_cat[cat]['correct'] += 1
        correct_count += 1
        continue

    if not evidence_set:
        failure_by_cat[cat]['no_evidence'] += 1
        failures['no_evidence'].append((q, cat, conv_id))
        continue

    # Check retrieval
    found_at_20 = len(evidence_set & set(retrieved_ids[:20]))
    if found_at_20 == 0:
        failure_by_cat[cat]['retrieval'] += 1
        failures['retrieval'].append((q, answer, cat, conv_id, evidence))
    elif found_at_20 < len(evidence_set):
        failure_by_cat[cat]['partial'] += 1
        failures['partial'].append((q, answer, cat, conv_id, evidence, found_at_20, len(evidence_set)))
    else:
        failure_by_cat[cat]['generation'] += 1
        failures['generation'].append((q, answer, generated, cat, conv_id))

print(f"\n{'='*70}")
print(f"FAILURE ANALYSIS RESULTS ({total} questions)")
print(f"{'='*70}")
print(f"\nOverall: {correct_count}/{total} correct ({correct_count/total*100:.1f}%)")

n_fail = total - correct_count
print(f"Failures: {n_fail}")

r = len(failures['retrieval'])
p = len(failures['partial'])
g = len(failures['generation'])
ne = len(failures['no_evidence'])
if n_fail > 0:
    print(f"\n  Retrieval failure (R@20=0):  {r:3d} ({r/n_fail*100:.0f}%)")
    print(f"  Partial retrieval:           {p:3d} ({p/n_fail*100:.0f}%)")
    print(f"  Generation/ranking failure:  {g:3d} ({g/n_fail*100:.0f}%)")
    print(f"  No evidence IDs:             {ne:3d} ({ne/n_fail*100:.0f}%)")

print(f"\n{'='*70}")
print(f"PER-CATEGORY FAILURE BREAKDOWN")
print(f"{'='*70}")
header = f"{'Category':15s} {'Total':>5s} {'Correct':>10s} {'Retrieval':>12s} {'Partial':>10s} {'Gen/Rank':>11s} {'NoEvid':>8s}"
print(header)
print("-" * len(header))
for cat in sorted(cats):
    d = failure_by_cat[cat]
    t = sum(d.values())
    if t == 0:
        continue
    n_wrong = t - d['correct']
    print(f"{cats[cat]:15s} {t:5d} {d['correct']:4d} ({d['correct']/t*100:4.0f}%) "
          f"{d['retrieval']:4d} ({d['retrieval']/max(1,n_wrong)*100:3.0f}%) "
          f"{d['partial']:4d} ({d['partial']/max(1,n_wrong)*100:3.0f}%) "
          f"{d['generation']:4d} ({d['generation']/max(1,n_wrong)*100:3.0f}%) "
          f"{d['no_evidence']:4d}")

# Sample failures
print(f"\n{'='*70}")
print(f"SAMPLE RETRIEVAL FAILURES (first 15)")
print(f"{'='*70}")
for q, ans, cat, conv_id, ev in failures['retrieval'][:15]:
    print(f"  [{cats[cat]}] {conv_id}")
    print(f"    Q: {q[:90]}")
    print(f"    A: {str(ans)[:90]}")
    print(f"    Evidence: {ev[:3]}")
    print()

print(f"\n{'='*70}")
print(f"SAMPLE GENERATION FAILURES (first 15)")
print(f"{'='*70}")
for item in failures['generation'][:15]:
    q, ans, gen, cat, conv_id = item
    print(f"  [{cats[cat]}] {conv_id}")
    print(f"    Q: {q[:90]}")
    print(f"    Expected: {str(ans)[:90]}")
    print(f"    Got:      {gen[:90]}")
    print()

print(f"\n{'='*70}")
print(f"SAMPLE PARTIAL RETRIEVAL FAILURES (first 10)")
print(f"{'='*70}")
for item in failures['partial'][:10]:
    q, ans, cat, conv_id, ev, found, total_ev = item
    print(f"  [{cats[cat]}] {conv_id} — found {found}/{total_ev} evidence")
    print(f"    Q: {q[:90]}")
    print(f"    A: {str(ans)[:90]}")
    print()
