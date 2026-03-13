# LongMemEval Evaluation

Evaluates MnemeFusion on [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (ICLR 2025) — a benchmark for long-term conversational memory systems.

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

- **Oracle**: Each question gets only the sessions containing evidence (~36 turns). Tests extraction + RAG quality without retrieval noise. Good for development.
- **Full haystack (s)**: Each question gets all ~490 turns. Tests end-to-end retrieval. Required for publication.

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

# Full haystack mode (for publication)
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
| `--mode {oracle,s}` | Dataset mode: oracle (evidence-only) or s (full haystack) |
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

Results are saved incrementally to JSON — the script is crash-safe and resumable.

## Metrics

- **Task-averaged accuracy**: Mean of the 6 per-category accuracies (each category weighted equally). Primary metric.
- **Overall accuracy**: Mean across all individual binary labels (instance-weighted).
- **Abstention accuracy**: Accuracy on questions where the correct answer is "no information available".
- **Recall@K**: Fraction of gold evidence turns found in top-K retrieved memories.

## Output

Results are saved to `longmemeval_results_{mode}_{binary|detailed}.json` with per-question scores, retrieval recall, and category breakdowns.

## Results

Evaluated on the full LongMemEval dataset (500 questions, 6 categories) with Phi-4-mini-instruct (Q4_K_M, 2.5GB) for entity extraction on an NVIDIA A40 GPU.

### Oracle Mode (evidence-only context)

Each question receives only the sessions containing gold evidence (~36 turns). Tests extraction + RAG quality without retrieval noise.

| Metric | Score |
|--------|-------|
| **Task-averaged accuracy** | **90.9%** |
| **Overall accuracy** | **90.0%** |

| Category | Count | Accuracy |
|----------|-------|----------|
| single-session-assistant | 56 | 100.0% |
| single-session-user | 70 | 98.6% |
| knowledge-update | 78 | 91.0% |
| temporal-reasoning | 133 | 88.0% |
| multi-session | 133 | 84.2% |
| single-session-preference | 30 | 83.3% |

Retrieval: R@5=68.6%, R@10=82.5%, R@20=93.7%.

### S-Mode (full haystack — ~490 turns per question)

Each question gets ALL conversation turns (~490), requiring end-to-end retrieval from a large haystack. Per-question fresh ingestion with LLM entity extraction.

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

### Oracle vs S-Mode Gap

The 53-point gap is overwhelmingly a retrieval problem:

- **48.4%** of failed questions had zero gold evidence in top-20 results
- **49.0%** had partial evidence (some turns found, critical ones missing)
- **Only 2.5%** were reasoning failures (evidence retrieved but wrong answer)

The oracle result (90%) confirms the extraction + RAG + judge pipeline works. The s-mode result (37.2%) reflects the retrieval ceiling when a 3.8B model must extract searchable metadata from 490 turns — better extraction models will directly improve s-mode without architecture changes.

## References

- LongMemEval paper: Di Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
- Dataset: [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
