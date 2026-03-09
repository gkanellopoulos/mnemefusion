# Evaluations

MnemeFusion is evaluated on two established conversational memory benchmarks using standard protocols that produce numbers comparable to published results.

## Benchmarks

| Benchmark | Protocol | Questions | MnemeFusion | Status |
|-----------|----------|-----------|-------------|--------|
| [LoCoMo](locomo/) (standard) | Free-text + LLM-as-judge (Mem0-compatible) | 1,540 (cat 1-4) | **70.7% ± 0.8%** | Verified |
| [LoCoMo](locomo/) (atomized) | Same protocol, per-entity DB | 1,540 (cat 1-4) | **72.3% ± 0.1%** | Verified |
| [LongMemEval](longmemeval/) | Binary yes/no judge (official paper protocol) | 500 | 90.0% (oracle) | S-mode in progress |

## Methodology

Both benchmarks follow the protocols used by published systems (Mem0, Zep, MemMachine, Letta) to ensure results are directly comparable.

### LoCoMo

- **Answer generation**: GPT-4o-mini, temperature=0, constrained to 5-6 words
- **Judge**: GPT-4o-mini, temperature=0, binary CORRECT/WRONG with generous matching
- **Categories**: 1-4 (single-hop, multi-hop, temporal, open-domain) — 1,540 questions
- **Multi-run**: `--runs N` for mean ± stddev (recommended: 3 runs for publication)
- **MCQ mode**: Also available as `--mcq` for deterministic internal testing (non-standard)

### LongMemEval

- **Answer generation**: GPT-5-mini (our choice — clearly reported)
- **Judge**: gpt-4o-2024-08-06, temperature=0, binary yes/no (official paper requirement)
- **Prompts**: 5 task-specific prompts + 1 abstention prompt (matching official code)
- **Metrics**: Task-averaged accuracy (primary) + Overall accuracy (secondary)
- **Modes**: Oracle (development) and S (publication — full haystack retrieval)

## Reproducing Results

Both evaluations require:
1. MnemeFusion installed (`maturin develop --release`)
2. An embedding model (`BAAI/bge-base-en-v1.5`, auto-downloaded)
3. An OpenAI API key (for answer generation and judging)
4. Optionally, a GGUF model for local entity extraction

See each benchmark's README for detailed instructions.
