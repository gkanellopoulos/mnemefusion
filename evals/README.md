# Evaluations

MnemeFusion is evaluated on two established conversational memory benchmarks.

## Benchmarks

| Benchmark | Metric | Score | Questions | Status |
|-----------|--------|-------|-----------|--------|
| [LoCoMo](locomo/) | MCQ Accuracy | **70.2%** | 1,986 | Full evaluation |
| [LongMemEval](longmemeval/) | Avg Score | **95.1/100** | 65/500 | Partial (oracle mode) |

## LoCoMo

**Long-term Conversation Memory** — Tests retrieval and QA over 10 multi-session conversations with ~2000 questions across 5 categories (single-hop, multi-hop, temporal, open-domain, adversarial).

- [Full results and instructions](locomo/README.md)
- [Run evaluation](locomo/run_eval.py)

## LongMemEval

**Long-term Interactive Memory** (ICLR 2025) — Tests 6 categories of conversational memory: temporal reasoning, knowledge updates, preferences, single-session recall, and multi-session reasoning.

- [Full results and instructions](longmemeval/README.md)
- [Run evaluation](longmemeval/run_eval.py)

## Reproducing Results

Both evaluations require:
1. MnemeFusion installed (`maturin develop --release`)
2. An embedding model (`BAAI/bge-base-en-v1.5`, auto-downloaded)
3. An OpenAI API key (for answer generation/judging)
4. Optionally, a GGUF model for local entity extraction

See each benchmark's README for detailed instructions.
