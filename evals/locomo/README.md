# LoCoMo Evaluation

Evaluates MnemeFusion on the [LoCoMo](https://github.com/snap-research/locomo) (Long-term Conversation Memory) benchmark using the standard free-text + LLM-as-judge protocol.

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

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--db-path PATH` | Path to .mfdb database file |
| `--skip-ingestion` | Skip ingestion, use existing database |
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
| **Overall accuracy** | **70.7% ± 0.8%** |
| Per-run accuracies | 71.2%, 71.2%, 69.8% |

### Per-Category Breakdown

| Category | Count | Accuracy | R@5 | R@10 | R@20 |
|----------|-------|----------|-----|------|------|
| Single-hop (factual) | 282 | 64.5% | 14% | 21% | 33% |
| Multi-hop (reasoning) | 321 | 59.2% | 44% | 49% | 57% |
| Temporal (time-based) | 96 | 63.5% | 21% | 27% | 36% |
| Open-domain (knowledge) | 841 | 76.3% | 40% | 48% | 62% |

### Retrieval & Performance

| Metric | Value |
|--------|-------|
| Recall@5 | 35.0% |
| Recall@10 | 42.2% |
| Recall@20 | 54.0% |
| Latency P50 | 194ms |
| Latency P95 | 239ms |
| Ingestion time | ~1h47m (5,882 docs on A40) |

## Atomized Results (Per-Entity DB)

MnemeFusion is designed for one `.mfdb` file per entity — matching how memory works in production (one DB per user/customer/project). The atomized benchmark creates one DB per conversation, eliminating cross-entity retrieval noise.

**Setup:** Same as standard, but 10 separate DBs instead of 1 shared DB.

| Metric | Standard (shared DB) | Atomized (per-entity DB) | Delta |
|--------|---------------------|-------------------------|-------|
| **Overall accuracy** | **70.7% ± 0.8%** | **72.3% ± 0.1%** | **+1.6** |
| Single-hop (factual) | 64.5% | 72.0% | +7.5 |
| Multi-hop (reasoning) | 59.2% | 57.3% | -1.9 |
| Temporal (time-based) | 63.5% | 71.9% | +8.4 |
| Open-domain (knowledge) | 76.3% | 78.2% | +1.9 |
| Recall@20 | 54.0% | 56.2% | +2.2 |

Entity isolation lifts single-hop (+7.5) and temporal (+8.4) significantly — the categories most affected by cross-entity confusion in a shared DB.

## References

- LoCoMo dataset: [snap-research/locomo](https://github.com/snap-research/locomo)
- MCQ variant: [Percena/locomo-mc10](https://huggingface.co/datasets/Percena/locomo-mc10)
