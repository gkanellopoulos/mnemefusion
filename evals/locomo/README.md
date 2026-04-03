# LoCoMo Evaluation

Evaluates MnemeFusion on the [LoCoMo](https://github.com/snap-research/locomo) (Long-term Conversation Memory) benchmark using the standard free-text + LLM-as-judge protocol.

LoCoMo tests overall conversational memory quality across 10 multi-session conversations. All conversations are pre-ingested into a single database, measuring how well MnemeFusion retrieves and reasons over a realistic memory store.

## Protocol

| Aspect | Configuration |
|--------|---------------|
| Answer model | GPT-4o-mini, temperature=0 |
| Answer constraint | 5-6 words |
| Judge model | GPT-4o-mini, temperature=0 |
| Judge format | Binary CORRECT/WRONG with generous matching |
| Categories | 1-4 (single-hop, multi-hop, temporal, open-domain) |
| Questions | 1,540 |
| Multi-run | `--runs N` for mean ± stddev |

## Dataset

The LoCoMo dataset is included in this directory (`locomo10.json`, 2.8MB). It contains 10 multi-session conversations with ~2000 questions across 5 categories.

For MCQ evaluation (supplementary), you also need the [Percena/locomo-mc10](https://huggingface.co/datasets/Percena/locomo-mc10) dataset:

```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('Percena/locomo-mc10')"
```

## Prerequisites

```bash
# Install MnemeFusion
cd mnemefusion-python
maturin develop --release --features entity-extraction  # or entity-extraction-cuda for GPU

# Install Python dependencies
pip install sentence-transformers openai datasets

# Set OpenAI API key
export OPENAI_API_KEY=sk-...
```

## Running the Evaluation

### Standard Protocol (recommended)

Free-text generation + LLM-as-judge, categories 1-4:

```bash
# With pre-ingested database
python run_eval.py \
    --db-path <path-to.mfdb> \
    --skip-ingestion

# Multi-run for publication
python run_eval.py \
    --db-path <path-to.mfdb> \
    --skip-ingestion \
    --runs 3
```

### MCQ Mode (supplementary)

Deterministic 10-choice MCQ — useful for internal development, not directly comparable to published numbers:

```bash
python run_eval.py \
    --db-path <path-to.mfdb> \
    --skip-ingestion \
    --mcq
```

### Full Ingestion + Evaluation

Ingests all conversations with LLM entity extraction, then evaluates:

```bash
python run_eval.py \
    --use-llm \
    --llm-model <path-to-model.gguf> \
    --extraction-passes 3
```

**Note:** Full ingestion takes ~15-30 hours on a GTX 1650 Ti (GPU). The script supports checkpoint/resume.

### Quick Test (subset)

```bash
python run_eval.py \
    --db-path <path-to.mfdb> \
    --skip-ingestion \
    --max-questions 50
```

### Retrieval-Only Mode

Measure retrieval quality (R@5, R@10, R@20, MRR) without any LLM calls — no OpenAI API key needed:

```bash
python run_eval.py \
    --db-path <path-to.mfdb> \
    --skip-ingestion \
    --retrieval-only
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--db-path PATH` | Path to .mfdb database file |
| `--skip-ingestion` | Skip ingestion, use existing database |
| `--retrieval-only` | Retrieval metrics only (no LLM, no API key needed) |
| `--mcq` | MCQ mode (deterministic, non-standard) |
| `--runs N` | Number of runs (default: 1, recommend 3 for publication) |
| `--use-llm` | Enable LLM entity extraction during ingestion |
| `--llm-model PATH` | Path to GGUF model for extraction |
| `--extraction-passes N` | Number of extraction passes (1-3) |
| `--max-questions N` | Limit number of questions |
| `--categories 1 2 3` | Evaluate specific categories only |
| `--verbose` | Print per-question results |

## Output

The script prints a methodology header, per-category breakdown, recall@K metrics, and latency percentiles. Results can be saved to JSON with `--output path.json`.

## Results

Evaluated on the full LoCoMo dataset (10 conversations, 1,540 questions, categories 1-4) with 3 independent runs.

**Setup:** Phi-4-mini-instruct (Q4_K_M, 2.5GB) for entity extraction, NVIDIA A40 GPU, single-pass ingestion.

| Metric | Score |
|--------|-------|
| **Overall accuracy** | **69.9% ± 0.4%** |
| Per-run accuracies | 69.6%, 70.3%, 69.7% |

### Per-Category Breakdown

| Category | Count | Accuracy | R@5 | R@10 | R@20 |
|----------|-------|----------|-----|------|------|
| Single-hop (factual) | 282 | 64.9% | 18% | 23% | 37% |
| Multi-hop (reasoning) | 321 | 58.3% | 45% | 51% | 60% |
| Temporal (time-based) | 96 | 63.5% | 17% | 27% | 36% |
| Open-domain (knowledge) | 841 | 76.3% | 42% | 50% | 62% |

### Retrieval & Performance

| Metric | Value |
|--------|-------|
| Recall@5 | 36.8% |
| Recall@10 | 44.0% |
| Recall@20 | 55.5% |
| Latency P50 | 253ms |
| Latency P95 | 329ms |
| Ingestion time | ~3h47m (5,882 docs on A40) |

## References

- LoCoMo dataset: [snap-research/locomo](https://github.com/snap-research/locomo)
- MCQ variant: [Percena/locomo-mc10](https://huggingface.co/datasets/Percena/locomo-mc10)
