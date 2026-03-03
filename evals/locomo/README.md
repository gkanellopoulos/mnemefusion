# LoCoMo Evaluation

Evaluates MnemeFusion on the [LoCoMo](https://github.com/snap-research/locomo) (Long-term Conversation Memory) benchmark.

## Current Results

| Metric | Score | Details |
|--------|-------|---------|
| **MCQ Accuracy** | **70.2%** | 1,986 questions, 10-choice multiple-choice |
| Recall@5 | 22.6% | Fraction of gold evidence in top-5 |
| Recall@10 | 29.6% | Fraction of gold evidence in top-10 |
| Recall@20 | 44.1% | Fraction of gold evidence in top-20 |
| Latency P50 | 48ms | Median query time |
| Latency P95 | 64ms | 95th percentile query time |

**By category:**

| Category | Accuracy | Questions |
|----------|----------|-----------|
| Single-hop | 59.1% | 662 |
| Multi-hop | 72.3% | 567 |
| Temporal | 65.5% | 87 |
| Open-domain | 83.1% | 453 |
| Adversarial | 61.3% | 217 |

*Evaluated on 10 fully-ingested conversations with Phi-4-mini extraction (3-pass).*

## Dataset

The LoCoMo dataset is included in this directory (`locomo10.json`, 2.8MB). It contains 10 multi-session conversations with ~2000 questions across 5 categories.

For MCQ evaluation, you also need the [Percena/locomo-mc10](https://huggingface.co/datasets/Percena/locomo-mc10) dataset:

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

# Set OpenAI API key (for MCQ answer selection)
export OPENAI_API_KEY=sk-...
```

## Running the Evaluation

### Option 1: MCQ with Pre-Ingested Database (fastest)

If you have a pre-ingested `.mfdb` database:

```bash
python run_eval.py \
    --db-path <path-to.mfdb> \
    --skip-ingestion \
    --mcq
```

### Option 2: Full Ingestion + MCQ Evaluation

Ingests all conversations with LLM entity extraction, then evaluates:

```bash
python run_eval.py \
    --use-llm \
    --llm-model <path-to-model.gguf> \
    --extraction-passes 3 \
    --mcq
```

**Note:** Full ingestion takes ~15-30 hours on a GTX 1650 Ti (GPU) or longer on CPU. The script supports checkpoint/resume — if interrupted, re-running with `--db-path` will resume from where it left off.

### Option 3: Quick Test (subset)

```bash
python run_eval.py \
    --db-path <path-to.mfdb> \
    --skip-ingestion \
    --mcq \
    --max-questions 100
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--db-path PATH` | Path to .mfdb database file |
| `--skip-ingestion` | Skip ingestion, use existing database |
| `--mcq` | MCQ mode (deterministic, recommended) |
| `--use-llm` | Enable LLM entity extraction during ingestion |
| `--llm-model PATH` | Path to GGUF model for extraction |
| `--extraction-passes N` | Number of extraction passes (1-3) |
| `--max-questions N` | Limit number of questions |
| `--categories 1 2 3` | Evaluate specific categories only |
| `--num-conversations N` | Ingest only first N conversations |

## Output

Results are saved to a JSON file in `evals/locomo/` with per-question scores, retrieval metrics, and category breakdowns.

## Evaluation Methodology

- **MCQ mode** (recommended): Each question has 10 answer choices. GPT-4o-mini selects the best answer given retrieved context. Deterministic — no LLM judge variance.
- **Free-text mode**: GPT-4o-mini generates an answer, then a separate LLM judge scores correctness. Subject to ±1-2% judge noise.
- **Recall@K**: Measures what fraction of gold evidence turns appear in the top-K retrieved memories (substring matching).

## References

- LoCoMo dataset: [snap-research/locomo](https://github.com/snap-research/locomo)
- MCQ variant: [Percena/locomo-mc10](https://huggingface.co/datasets/Percena/locomo-mc10)
