# Evaluations

MnemeFusion is evaluated on two established conversational memory benchmarks using standard protocols. Together, the results validate the **atomic architecture** — per-entity databases maintain accuracy where a shared database degrades.

## Results

| Benchmark | Mode | What it tests | Score | Status |
|-----------|------|---------------|-------|--------|
| [LoCoMo](locomo/) | Standard | Overall accuracy across 10 conversations | **69.9% ± 0.4%** | Verified |
| [LongMemEval](longmemeval/) | Oracle | Pipeline quality (extraction + RAG + scoring) | **91.4%** | Verified |
| [LongMemEval](longmemeval/) | Per-entity | Production pattern: one DB per conversation | **67.6%** | Verified |
| [LongMemEval](longmemeval/) | Shared DB | All conversations in one DB | 37.2% | Verified |

### How to read these results

- **Oracle (91.4%)** gives each question only the sessions containing evidence, stripping away retrieval noise. This proves the extraction + RAG + judge pipeline works.
- **Per-entity (67.6%)** gives each question its own database with all ~490 conversation turns — the recommended atomic pattern. Each conversation maps to one entity's memory, testing end-to-end retrieval from a realistic haystack.
- **Shared DB (37.2%)** puts all conversations into a single database. The 91-to-37% collapse demonstrates why the per-entity architecture matters: unrelated memories flood retrieval results when they share a database.
- **LoCoMo (69.9%)** evaluates conversational memory across 1,540 questions and 10 conversations with free-text answers judged by GPT-4o-mini. Each conversation is pre-ingested into a shared database — a harder setup that measures general retrieval quality.

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
- **Modes**: Oracle (evidence-only), Per-entity (one DB per conversation), Shared DB (all conversations in one DB)

## Reproducing Results

Both evaluations require:
1. MnemeFusion installed (`maturin develop --release`)
2. An embedding model (`BAAI/bge-base-en-v1.5`, auto-downloaded)
3. An OpenAI API key (for answer generation and judging)
4. Optionally, a GGUF model for local entity extraction

See each benchmark's README for detailed instructions.
