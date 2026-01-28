#!/usr/bin/env python3
"""
Sprint 17 Ablation Study

Systematically measures the contribution of each component:
1. Baseline (semantic only)
2. + BM25 keyword search
3. + RRF fusion
4. + Graph traversal
5. + Heuristic reranking
6. + Multi-turn aggregation

Runs on both LoCoMo and LongMemEval to measure generalization.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List

class AblationStudy:
    def __init__(self):
        self.results = {
            'configurations': [],
            'locomo_results': [],
            'longmemeval_results': []
        }

    def run_configuration(self, config_name: str, description: str) -> Dict:
        """Run a single configuration on both benchmarks"""
        print(f"\n{'='*70}")
        print(f"Configuration: {config_name}")
        print(f"Description: {description}")
        print(f"{'='*70}\n")

        result = {
            'name': config_name,
            'description': description,
            'locomo': None,
            'longmemeval': None
        }

        # Run LoCoMo Phase 2
        print("Running LoCoMo Phase 2 benchmark...")
        try:
            subprocess.run([
                sys.executable,
                "tests/benchmarks/locomo_eval.py",
                "--samples", "10"
            ], check=True, capture_output=True, text=True, timeout=300)

            with open("tests/benchmarks/fixtures/locomo_phase2_results.json", 'r') as f:
                locomo_data = json.load(f)
                result['locomo'] = {
                    'recall': locomo_data['recall_at_k'],
                    'mrr': locomo_data['mrr'],
                    'categories': {
                        str(k): v['recall_at_k']
                        for k, v in locomo_data['category_metrics'].items()
                    }
                }
            print(f"  LoCoMo Recall: {result['locomo']['recall']:.1%}")
        except Exception as e:
            print(f"  ERROR running LoCoMo: {e}")
            result['locomo'] = {'error': str(e)}

        # Run LongMemEval Oracle
        print("Running LongMemEval Oracle benchmark...")
        try:
            subprocess.run([
                sys.executable,
                "tests/benchmarks/longmemeval_eval.py",
                "--variant", "oracle"
            ], check=True, capture_output=True, text=True, timeout=300)

            with open("tests/benchmarks/fixtures/longmemeval_oracle_results.json", 'r') as f:
                longmem_data = json.load(f)
                result['longmemeval'] = {
                    'recall': longmem_data['recall'],
                    'precision': longmem_data['precision'],
                    'f1': longmem_data['f1'],
                    'abilities': {
                        k: v['recall']
                        for k, v in longmem_data['ability_metrics'].items()
                    }
                }
            print(f"  LongMemEval Recall: {result['longmemeval']['recall']:.1%}")
            print(f"  LongMemEval F1: {result['longmemeval']['f1']:.3f}")
        except Exception as e:
            print(f"  ERROR running LongMemEval: {e}")
            result['longmemeval'] = {'error': str(e)}

        return result

    def run_study(self):
        """Run full ablation study"""
        print("\n" + "="*70)
        print("SPRINT 17 ABLATION STUDY")
        print("="*70)
        print("\nThis study will test 6 configurations on 2 benchmarks")
        print("Estimated time: ~30 minutes\n")

        configurations = [
            ("baseline", "Semantic search only (no BM25, no fusion, no graphs)"),
            ("bm25", "Baseline + BM25 keyword search"),
            ("rrf", "Baseline + BM25 + RRF fusion"),
            ("graph", "Baseline + BM25 + RRF + Graph traversal"),
            ("rerank", "Baseline + BM25 + RRF + Graph + Reranking"),
            ("full", "All components (current Sprint 17 system)")
        ]

        for config_name, description in configurations:
            result = self.run_configuration(config_name, description)
            self.results['configurations'].append(result)

            # Save incremental results
            self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save results to JSON file"""
        output_path = "tests/benchmarks/fixtures/ablation_study_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n[Saved incremental results to {output_path}]")

    def print_summary(self):
        """Print summary table of all configurations"""
        print("\n" + "="*70)
        print("ABLATION STUDY SUMMARY")
        print("="*70)
        print("\nLoCoMo Phase 2 Results:")
        print(f"{'Configuration':<20} {'Recall':>10} {'Cat 1':>8} {'Cat 3':>8} {'Cat 5':>8}")
        print("-" * 70)

        for config in self.results['configurations']:
            if config['locomo'] and 'recall' in config['locomo']:
                recall = config['locomo']['recall']
                cat1 = config['locomo']['categories'].get('1', 0)
                cat3 = config['locomo']['categories'].get('3', 0)
                cat5 = config['locomo']['categories'].get('5', 0)
                print(f"{config['name']:<20} {recall:>9.1%} {cat1:>7.1%} {cat3:>7.1%} {cat5:>7.1%}")

        print("\n\nLongMemEval Oracle Results:")
        print(f"{'Configuration':<20} {'Recall':>10} {'Precision':>10} {'F1':>10}")
        print("-" * 70)

        for config in self.results['configurations']:
            if config['longmemeval'] and 'recall' in config['longmemeval']:
                recall = config['longmemeval']['recall']
                precision = config['longmemeval']['precision']
                f1 = config['longmemeval']['f1']
                print(f"{config['name']:<20} {recall:>9.1%} {precision:>9.1%} {f1:>9.3f}")

        print("\n" + "="*70)

if __name__ == "__main__":
    print("\n⚠️  WARNING: This script requires code modifications to disable components.")
    print("The current implementation will run the full system for all configurations.")
    print("To properly run ablation, you need to modify QueryPlanner to accept config flags.\n")

    response = input("Continue anyway for testing? (y/n): ")
    if response.lower() != 'y':
        print("Ablation study cancelled.")
        sys.exit(0)

    study = AblationStudy()
    study.run_study()
