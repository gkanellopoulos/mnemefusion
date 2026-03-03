# LongMemEval Evaluation

Evaluates MnemeFusion on [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (ICLR 2025) — a benchmark for long-term conversational memory systems.

## Current Results (Oracle Mode)

| Category | n | Avg Score | Pass Rate (>=80) |
|----------|---|-----------|-------------------|
| temporal-reasoning | 10 | 100.0 | 100% |
| single-session-assistant | 15 | 99.9 | 100% |
| knowledge-update | 10 | 96.5 | 100% |
| single-session-preference | 10 | 96.3 | 100% |
| single-session-user | 10 | 89.3 | 90% |
| multi-session | 10 | 86.5 | 80% |
| **Overall** | **65** | **95.1** | **95.4%** |

**Retrieval metrics:**

| Metric | Score |
|--------|-------|
| Recall@5 | 68.6% |
| Recall@10 | 88.2% |
| Recall@20 | 96.5% |

*Oracle mode evaluates 65/500 questions (10+ per category) with evidence-only sessions. Full evaluation pending.*

## Dataset

LongMemEval is too large (~280MB) to include in the repository. Download from HuggingFace:

```bash
pip install datasets

python -c "
from datasets import load_dataset
ds = load_dataset('xiaowu0162/longmemeval-cleaned')

# Save oracle split (evidence-only, ~15MB)
import json
with open('evals/longmemeval/longmemeval_oracle.json', 'w') as f:
    json.dump([dict(row) for row in ds['oracle']], f)

# Save full split (full haystack, ~265MB)
with open('evals/longmemeval/longmemeval_s.json', 'w') as f:
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

# Set OpenAI API key (for RAG answer generation + judging)
export OPENAI_API_KEY=sk-...
```

## Running the Evaluation

### Oracle Mode (recommended first)

Oracle mode uses evidence-only sessions (~36 turns per question). This validates that extraction and RAG work correctly without retrieval noise.

```bash
python run_eval.py \
    --mode oracle \
    --llm-model <path-to-model.gguf>
```

### Full Haystack Mode

Full haystack uses all conversation turns (~490 per question), testing end-to-end retrieval:

```bash
python run_eval.py \
    --mode s \
    --llm-model <path-to-model.gguf>
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--mode {oracle,s}` | Evaluation mode (oracle = evidence-only, s = full haystack) |
| `--llm-model PATH` | Path to GGUF model for entity extraction |
| `--start-at N` | Resume from question N (0-indexed) |
| `--max-questions N` | Stop after N new questions |
| `--category NAME` | Only evaluate a specific category |
| `--extraction-passes N` | Number of extraction passes (default: 1) |
| `--stop-on-fail` | Stop on first question scoring <50 |

## How It Works

Each question is evaluated independently:

1. **Ingest**: All conversation sessions for the question are ingested into a fresh `.mfdb` with LLM entity extraction
2. **Query**: The question is embedded and queried against the database (top-20 retrieval)
3. **Answer**: GPT-5-mini generates an answer from retrieved context
4. **Judge**: GPT-5-mini scores the answer on a 0-100 scale against the gold answer
5. **Recall**: Gold evidence turns are matched against retrieved memories

Results are saved incrementally to JSON — the script is crash-safe and resumable.

## Output

Results are saved to `longmemeval_{mode}_results.json` with per-question scores, retrieval recall, and category breakdowns.

## Evaluation Modes Explained

- **Oracle**: Each question gets only the sessions containing evidence. Eliminates retrieval difficulty — if the system fails here, the problem is in extraction or RAG, not retrieval. Fast (~5 min/question).
- **Full haystack (s)**: Each question gets all ~490 turns. Tests the full retrieval pipeline including vector search, entity scoring, and RRF fusion. Slow (~75 min/question with LLM extraction).

## References

- LongMemEval paper: Di Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
- Dataset: [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
