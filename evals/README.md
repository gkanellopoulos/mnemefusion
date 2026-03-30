# Evaluations

MnemeFusion is evaluated on two established conversational memory benchmarks using standard protocols.

## Benchmarks

| Benchmark | Protocol | Questions | MnemeFusion | Status |
|-----------|----------|-----------|-------------|--------|
| [LoCoMo](locomo/) (standard) | Free-text + LLM-as-judge | 1,540 (cat 1-4) | **70.7% ± 0.8%** | Verified |
| [LoCoMo](locomo/) (atomized) | Per-entity DB, same protocol | 1,540 (cat 1-4) | **72.3% ± 0.1%** | Verified |
| [LongMemEval](longmemeval/) (oracle) | Binary judge (official protocol) | 500 | **90.0%** | Verified |
| [LongMemEval](longmemeval/) (s-mode) | Full haystack, same protocol | 500 | **37.2%** | Verified |
| [LongMemEval](longmemeval/) (atomic) | Per-entity DB, binary judge | 176 | **64.8%** | Verified |

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
- **Modes**: Oracle (evidence-only, for development) and S (full haystack, for publication)

## Reproducing Results

Both evaluations require:
1. MnemeFusion installed (`maturin develop --release`)
2. An embedding model (`BAAI/bge-base-en-v1.5`, auto-downloaded)
3. An OpenAI API key (for answer generation and judging)
4. Optionally, a GGUF model for local entity extraction

See each benchmark's README for detailed instructions.
