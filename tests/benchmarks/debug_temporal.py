#!/usr/bin/env python3
"""Debug temporal range extraction"""

import json

# Load dataset
with open('fixtures/locomo10.json') as f:
    data = json.load(f)

# Extract some Category 2 queries (temporal)
category_2_queries = []
for conv in data:
    sample_id = conv['sample_id']
    for qa in conv['qa']:
        if qa.get('category') == 2:
            category_2_queries.append({
                'sample_id': sample_id,
                'question': qa['question'],
                'category': qa['category']
            })
            if len(category_2_queries) >= 20:
                break
    if len(category_2_queries) >= 20:
        break

print("=== Category 2 (Temporal) Query Examples ===\n")
for i, qa in enumerate(category_2_queries[:20], 1):
    print(f"{i}. {qa['question']}")
    print()
