# LongMemEval Evaluation

Evaluates MnemeFusion on [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (ICLR 2025) — a benchmark for long-term conversational memory systems.

## Evaluation Protocol

| Aspect | Configuration |
|--------|---------------|
| Answer model | GPT-5-mini (our choice, clearly reported) |
| Judge model | gpt-4o-2024-08-06 (official paper requirement) |
| Scoring | Binary yes/no (official protocol) |
| Judge prompts | 5 task-specific + 1 abstention (matching official code) |
| Metrics | Task-averaged accuracy (primary), Overall accuracy (secondary) |
| Temporal | Off-by-one forgiveness for temporal-reasoning |
| Knowledge update | Accepts old+updated or just updated answer |
| Preference | Lenient matching for preference questions |

This protocol matches the official LongMemEval evaluation code, which asserts `model == 'gpt-4o-2024-08-06'` for the judge. Using any other model makes results non-comparable.

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

### Detailed Scoring (internal development)

For granular 0-100 scoring with GPT-5-mini (non-standard, for diagnosis):

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
| `--detailed-scoring` | Use 0-100 scoring with GPT-5-mini (non-standard, internal) |
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

- **Task-averaged accuracy**: Mean of the 6 per-category accuracies (each category weighted equally). This is the headline metric.
- **Overall accuracy**: Mean across all individual binary labels (instance-weighted).
- **Abstention accuracy**: Accuracy on questions where the correct answer is "no information available".
- **Recall@K**: Fraction of gold evidence turns found in top-K retrieved memories.

## Output

Results are saved to `longmemeval_results_{mode}_{binary|detailed}.json` with per-question scores, retrieval recall, and category breakdowns.

## References

- LongMemEval paper: Di Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
- Dataset: [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
