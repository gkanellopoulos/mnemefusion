# Evaluations

MnemeFusion is evaluated on two established conversational memory benchmarks using standard protocols.

## Benchmarks

| Benchmark | Protocol | Questions | MnemeFusion | Status |
|-----------|----------|-----------|-------------|--------|
| [LoCoMo](locomo/) (standard) | Free-text + LLM-as-judge | 1,540 (cat 1-4) | **69.9% ± 0.4%** | Verified |
| [LongMemEval](longmemeval/) (oracle) | Binary judge (official protocol) | 500 | **91.4%** | Verified |
| [LongMemEval](longmemeval/) (atomic) | Per-entity DB, full haystack | 176 | **67.6%** | Verified |

## Methodology

### LoCoMo

- **Answer generation**: GPT-4o-mini, temperature=0, constrained to 5-6 words
- **Judge**: GPT-4o-mini, temperature=0, binary CORRECT/WRONG with generous matching
- **Categories**: 1-4 (single-hop, multi-hop, temporal, open-domain) — 1,540 questions
- **Multi-run**: `--runs N` for mean ± stddev (recommended: 3 runs for publication)

### LongMemEval

- **Answer generation**: GPT-5-mini
- **Judge**: gpt-4o-2024-08-06, temperature=0, binary yes/no (official paper requirement)
- **Prompts**: 5 task-specific prompts + 1 abstention prompt (matching official code)
- **Metrics**: Task-averaged accuracy (primary) + Overall accuracy (secondary)
- **Modes**: Oracle (evidence-only), S (full haystack), Atomic (per-entity, 176 single-haystack questions)

## Reproducing Results

Both evaluations require:
1. MnemeFusion installed (`maturin develop --release`)
2. An embedding model (`BAAI/bge-base-en-v1.5`, auto-downloaded)
3. An OpenAI API key (for answer generation and judging)
4. Optionally, a GGUF model for local entity extraction

See each benchmark's README for detailed instructions.
