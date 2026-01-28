#!/usr/bin/env python3
"""
Analyze LoCoMo dataset to understand why Categories 1 and 3 perform poorly.
"""

import json
from collections import defaultdict
import random

def load_locomo_data(path):
    """Load and flatten LoCoMo dataset."""
    with open(path) as f:
        data = json.load(f)

    # Flatten all QA pairs with metadata
    all_qa = []
    for conv_id, conv in enumerate(data):
        for qa in conv['qa']:
            all_qa.append({
                'conv_id': conv_id,
                'question': qa['question'],
                'answer': qa.get('answer', 'N/A'),
                'evidence': qa['evidence'],
                'category': qa['category'],
                'sample_id': conv.get('sample_id', f'conv_{conv_id}')
            })

    return all_qa, data

def analyze_categories(all_qa):
    """Group QA pairs by category and compute statistics."""
    by_cat = defaultdict(list)
    for qa in all_qa:
        by_cat[qa['category']].append(qa)

    print("=" * 80)
    print("CATEGORY STATISTICS")
    print("=" * 80)
    for cat in sorted(by_cat.keys()):
        print(f"\nCategory {cat}: {len(by_cat[cat])} queries")

    return by_cat

def print_samples(category, queries, num_samples=10):
    """Print sample queries from a category."""
    cat_names = {
        1: "Factual",
        2: "Multi-hop",
        3: "Temporal reasoning",
        4: "Entity",
        5: "Contextual"
    }

    print("\n" + "=" * 80)
    print(f"CATEGORY {category}: {cat_names.get(category, 'Unknown')}")
    print(f"Total queries: {len(queries)}")
    print("=" * 80)

    # Sample queries
    samples = queries[:num_samples] if len(queries) >= num_samples else queries

    for i, qa in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Question: {qa['question']}")
        print(f"Answer: {qa['answer']}")
        print(f"Evidence: {', '.join(qa['evidence'])}")
        print(f"Conversation: {qa['sample_id']}")

def analyze_question_patterns(by_cat):
    """Analyze linguistic patterns in questions."""
    print("\n" + "=" * 80)
    print("QUESTION PATTERN ANALYSIS")
    print("=" * 80)

    for cat in sorted(by_cat.keys()):
        queries = by_cat[cat]

        # Extract question starters
        starters = defaultdict(int)
        for qa in queries:
            q = qa['question']
            first_word = q.split()[0].lower() if q else ''
            starters[first_word] += 1

        print(f"\nCategory {cat} question starters:")
        for starter, count in sorted(starters.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / len(queries)
            print(f"  {starter}: {count} ({pct:.1f}%)")

        # Average question length
        avg_len = sum(len(qa['question'].split()) for qa in queries) / len(queries)
        print(f"  Average question length: {avg_len:.1f} words")

        # Average answer length
        avg_ans_len = sum(len(str(qa['answer']).split()) for qa in queries) / len(queries)
        print(f"  Average answer length: {avg_ans_len:.1f} words")

def analyze_evidence_patterns(by_cat, conversations):
    """Analyze evidence patterns (which conversation turns are referenced)."""
    print("\n" + "=" * 80)
    print("EVIDENCE PATTERN ANALYSIS")
    print("=" * 80)

    for cat in sorted(by_cat.keys()):
        queries = by_cat[cat]

        # Count evidence references
        evidence_counts = []
        for qa in queries:
            evidence_counts.append(len(qa['evidence']))

        avg_evidence = sum(evidence_counts) / len(evidence_counts)
        single_evidence = sum(1 for c in evidence_counts if c == 1)
        multi_evidence = sum(1 for c in evidence_counts if c > 1)

        print(f"\nCategory {cat}:")
        print(f"  Average evidence references: {avg_evidence:.2f}")
        print(f"  Single evidence: {single_evidence} ({100*single_evidence/len(queries):.1f}%)")
        print(f"  Multi evidence: {multi_evidence} ({100*multi_evidence/len(queries):.1f}%)")

def compare_failing_vs_successful(by_cat):
    """Compare characteristics of failing vs successful categories."""
    print("\n" + "=" * 80)
    print("FAILING vs SUCCESSFUL CATEGORIES COMPARISON")
    print("=" * 80)

    failing = [1, 3]  # 16.0% and 21.6% recall
    successful = [2, 4, 5]  # 72.5%, 74.7%, 82.6% recall

    print("\nFailing categories (1, 3):")
    failing_queries = []
    for cat in failing:
        failing_queries.extend(by_cat[cat])

    print(f"  Total queries: {len(failing_queries)}")
    avg_len = sum(len(qa['question'].split()) for qa in failing_queries) / len(failing_queries)
    print(f"  Avg question length: {avg_len:.1f} words")

    print("\nSuccessful categories (2, 4, 5):")
    successful_queries = []
    for cat in successful:
        successful_queries.extend(by_cat[cat])

    print(f"  Total queries: {len(successful_queries)}")
    avg_len = sum(len(qa['question'].split()) for qa in successful_queries) / len(successful_queries)
    print(f"  Avg question length: {avg_len:.1f} words")

def main():
    dataset_path = 'locomo10.json'

    print("Loading LoCoMo dataset...")
    all_qa, conversations = load_locomo_data(dataset_path)

    print(f"\nTotal QA pairs: {len(all_qa)}")
    print(f"Total conversations: {len(conversations)}")

    # Group by category
    by_cat = analyze_categories(all_qa)

    # Print samples from each category
    # FAILING CATEGORIES (low recall)
    print_samples(1, by_cat[1], num_samples=10)
    print_samples(3, by_cat[3], num_samples=10)

    # SUCCESSFUL CATEGORIES (high recall)
    print_samples(2, by_cat[2], num_samples=8)
    print_samples(4, by_cat[4], num_samples=8)
    print_samples(5, by_cat[5], num_samples=8)

    # Analyze patterns
    analyze_question_patterns(by_cat)
    analyze_evidence_patterns(by_cat, conversations)
    compare_failing_vs_successful(by_cat)

if __name__ == '__main__':
    main()
