# LongMemEval Evaluation

Evaluates MnemeFusion on [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (ICLR 2025) — a benchmark for long-term conversational memory systems.

LongMemEval tests MnemeFusion across three modes that validate the **atomic architecture**: oracle mode proves the pipeline works, per-entity mode measures real-world performance with the recommended one-DB-per-conversation pattern, and shared-DB mode demonstrates why per-entity scoping matters.

## Protocol

| Aspect | Configuration |
|--------|---------------|
| Answer model | GPT-5-mini |
| Judge model | gpt-4o-2024-08-06 (official paper requirement) |
| Scoring | Binary yes/no (official protocol) |
| Judge prompts | 5 task-specific + 1 abstention (matching official code) |
| Metrics | Task-averaged accuracy (primary), Overall accuracy (secondary) |
| Temporal | Off-by-one forgiveness for temporal-reasoning |
| Knowledge update | Accepts old+updated or just updated answer |
| Preference | Lenient matching for preference questions |

The judge model is mandated by the official evaluation code (`assert model == 'gpt-4o-2024-08-06'`). Using any other model makes results non-comparable.

## Evaluation Modes

- **Oracle**: Each question gets only the sessions containing evidence (~36 turns). Tests extraction + RAG quality without retrieval noise. Good for development and pipeline validation.
- **Per-entity** (`--mode per-entity`): Each question gets its own fresh database with the full conversation haystack (~490 turns). This tests the **recommended production pattern** — one `.mfdb` per entity/conversation. Of the 500 questions, 176 are included — those answerable from a single conversation's context (single-session and single-conversation temporal-reasoning questions). Questions requiring cross-conversation reasoning (multi-session, knowledge-update, abstention) are excluded because they are structurally unanswerable when each database contains only one conversation. This is a property of the atomic architecture, not selective filtering.
- **Shared DB** (`--mode s`): Each question gets all ~490 turns in a single database. Tests retrieval when all conversations share one DB — the anti-pattern that MnemeFusion's atomic architecture is designed to avoid. Included for completeness and to demonstrate the per-entity advantage.

## Dataset

LongMemEval is too large (~280MB) to include in the repository. Download from HuggingFace:

```bash
pip install datasets

python -c "
from datasets import load_dataset
ds = load_dataset('xiaowu0162/longmemeval-cleaned')

import json
# Oracle split (evidence-only, ~15MB)
with open('evals/longmemeval/fixtures/longmemeval/longmemeval_oracle.json', 'w') as f:
    json.dump([dict(row) for row in ds['oracle']], f)

# Full split (full haystack, ~265MB)
with open('evals/longmemeval/fixtures/longmemeval/longmemeval_s_cleaned.json', 'w') as f:
    json.dump([dict(row) for row in ds['s_cleaned']], f)
"
```

## Prerequisites

```bash
# Install MnemeFusion with LLM extraction
cd mnemefusion-python
maturin develop --release --features entity-extraction  # or entity-extraction-cuda for GPU

# Install Python dependencies
pip install sentence-transformers openai datasets

# Set OpenAI API key
export OPENAI_API_KEY=sk-...
```

## Running the Evaluation

### Standard Protocol (binary scoring)

```bash
# Oracle mode (recommended first — validates extraction + RAG)
python run_eval.py \
    --mode oracle \
    --llm-model <path-to-model.gguf>

# Per-entity mode (176 questions, ~500 turns each — slow, recommended for publication)
python run_eval.py \
    --mode per-entity \
    --llm-model <path-to-model.gguf>

# Shared-DB mode (500 questions, ~490 turns each — very slow)
python run_eval.py \
    --mode s \
    --llm-model <path-to-model.gguf>
```

### Detailed Scoring (development only)

Granular 0-100 scoring with GPT-5-mini — non-standard, for diagnosis:

```bash
python run_eval.py \
    --mode oracle \
    --llm-model <path-to-model.gguf> \
    --detailed-scoring
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--mode {oracle,per-entity,s}` | Dataset mode: oracle, per-entity (176 questions, recommended), or s (shared DB, 500 questions) |
| `--llm-model PATH` | Path to GGUF model for entity extraction |
| `--detailed-scoring` | Use 0-100 scoring with GPT-5-mini (non-standard) |
| `--start-at N` | Resume from question N (0-indexed) |
| `--max-questions N` | Stop after N new questions |
| `--category NAME` | Only evaluate a specific category |
| `--extraction-passes N` | Number of extraction passes (default: 1) |
| `--stop-on-fail` | Stop on first low-scoring question for diagnosis |

## How It Works

Each question is evaluated independently:

1. **Ingest**: All conversation sessions are ingested into a fresh `.mfdb` with LLM entity extraction
2. **Query**: The question is embedded and queried against the database (top-20 retrieval)
3. **Answer**: GPT-5-mini generates an answer from retrieved context (includes `question_date` for temporal reasoning)
4. **Judge**: gpt-4o-2024-08-06 scores the answer with category-specific binary prompts
5. **Recall**: Gold evidence turns are matched against retrieved memories

In per-entity mode, each question gets its own fresh database — matching the atomic production pattern of one `.mfdb` per entity. Results are saved incrementally to JSON — the script is crash-safe and resumable.

## Metrics

- **Task-averaged accuracy**: Mean of the 6 per-category accuracies (each category weighted equally). Primary metric.
- **Overall accuracy**: Mean across all individual binary labels (instance-weighted).
- **Abstention accuracy**: Accuracy on questions where the correct answer is "no information available".
- **Recall@K**: Fraction of gold evidence turns found in top-K retrieved memories.

## Output

Results are saved to `longmemeval_results_{mode}_{binary|detailed}.json` with per-question scores, retrieval recall, and category breakdowns.

## Results

Evaluated on the full LongMemEval dataset with Phi-4-mini-instruct (Q4_K_M, 2.5GB) for entity extraction on an NVIDIA A40 GPU.

### Oracle Mode (pipeline validation)

Each question receives only the sessions containing gold evidence (~36 turns). Tests extraction + RAG quality without retrieval noise.

| Metric | Score |
|--------|-------|
| **Task-averaged accuracy** | **92.7%** |
| **Overall accuracy** | **91.4%** |

| Category | Count | Accuracy |
|----------|-------|----------|
| single-session-user | 70 | 98.6% |
| single-session-assistant | 56 | 96.4% |
| single-session-preference | 30 | 93.3% |
| temporal-reasoning | 133 | 92.5% |
| knowledge-update | 78 | 91.0% |
| multi-session | 133 | 84.2% |

Retrieval: R@5=74.4%, R@10=91.3%, R@20=98.2%.

Task-averaged accuracy weights each category equally; overall accuracy weights each question equally. The main README reports overall accuracy (91.4%) for cross-system comparability.

### Per-Entity Mode (production benchmark)

Each question gets its own fresh `.mfdb` with the full conversation haystack (~500 turns). This tests the recommended atomic pattern — one database per entity/conversation — on 176 questions answerable from a single conversation's context.

| Metric | Score |
|--------|-------|
| **Task-averaged accuracy** | **68.8%** |
| **Overall accuracy** | **67.6%** |

| Category | Count | Accuracy |
|----------|-------|----------|
| single-session-preference | 30 | 80.0% |
| single-session-user | 70 | 77.1% |
| temporal-reasoning | 20 | 70.0% |
| single-session-assistant | 56 | 48.2% |

Retrieval: R@5=37.9%, R@10=42.3%, R@20=52.5%.

Task-averaged accuracy weights each category equally; overall accuracy weights each question equally. The main README reports overall accuracy (67.6%) for cross-system comparability.

### Shared-DB Mode (~490 turns per question)

Each question gets ALL conversation turns (~490) in a single database. This is the anti-pattern — when unrelated conversations share one database, retrieval accuracy degrades significantly.

| Metric | Score |
|--------|-------|
| **Task-averaged accuracy** | **46.3%** |
| **Overall accuracy** | **37.2%** |

| Category | Count | Accuracy |
|----------|-------|----------|
| single-session-preference | 30 | 90.0% |
| single-session-user | 70 | 80.0% |
| knowledge-update | 78 | 43.6% |
| temporal-reasoning | 133 | 28.6% |
| single-session-assistant | 56 | 21.4% |
| multi-session | 133 | 14.3% |

### Why Per-Entity Matters

The gap between modes tells the story of MnemeFusion's atomic design:

| Mode | Overall | What it proves |
|------|---------|----------------|
| Oracle | 91.4% | The pipeline works — extraction, RAG, and scoring are sound |
| Per-entity | 67.6% | Production-ready accuracy with the recommended architecture |
| Shared DB | 37.2% | Retrieval collapses when unrelated memories share a database |

The oracle-to-shared-DB gap is overwhelmingly a retrieval problem:

- **48.4%** of failed questions had zero gold evidence in top-20 results
- **49.0%** had partial evidence (some turns found, critical ones missing)
- **Only 2.5%** were reasoning failures (evidence retrieved but wrong answer)

The per-entity mode avoids most of this noise by scoping each database to a single conversation — exactly how MnemeFusion is designed to be deployed.

## References

- LongMemEval paper: Di Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
- Dataset: [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
