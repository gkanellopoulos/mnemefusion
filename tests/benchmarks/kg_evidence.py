"""Find minimum documents needed for first 10 questions of conv 0."""
import json

with open('tests/benchmarks/fixtures/locomo10.json') as f:
    data = json.load(f)

qa = data[0]['qa']
conv = data[0]['conversation']

all_evidence = set()
for i, q in enumerate(qa[:10]):
    ev = q.get('evidence', [])
    all_evidence.update(ev)
    cat = q['category']
    print(f'[{i}] cat={cat} evidence={ev}')
    print(f'    Q: {q["question"]}')
    print(f'    A: {str(q["answer"])[:60]}')

print(f'\nAll evidence dia_ids: {sorted(all_evidence)}')
print(f'Total unique: {len(all_evidence)}')

# Map dia_ids to actual turns
print('\n--- Required Documents ---')
for did in sorted(all_evidence):
    # Parse D{session}:{turn} format
    parts = did.split(':')
    session = int(parts[0][1:])  # D1 -> 1
    turn_idx_1based = int(parts[1])

    turns = conv.get(f'session_{session}', [])
    # dia_id is 1-based in the dataset
    for t in turns:
        if t.get('dia_id') == did:
            print(f'  {did} | {t["speaker"]}: {t["text"][:100]}')
            break
    else:
        print(f'  {did} | NOT FOUND')
