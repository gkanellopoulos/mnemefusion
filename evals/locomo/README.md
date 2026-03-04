# LoCoMo Evaluation

Evaluates MnemeFusion on the [LoCoMo](https://github.com/snap-research/locomo) (Long-term Conversation Memory) benchmark using the standard free-text + LLM-as-judge protocol.

## Evaluation Protocol

| Aspect | Configuration |
|--------|---------------|
| Answer model | GPT-4o-mini, temperature=0 |
| Answer constraint | "5-6 words" (matching Mem0 protocol) |
| Judge model | GPT-4o-mini, temperature=0 |
| Judge format | Binary CORRECT/WRONG with generous matching |
| Categories | 1-4 (single-hop, multi-hop, temporal, open-domain) |
| Questions | 1,540 |
| Multi-run | `--runs N` for mean ± stddev |

This protocol matches the evaluation methodology used by Mem0, Zep, MemMachine, and other published systems, making results directly comparable.

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

## Evaluation Modes

- **Standard mode** (default): Free-text generation + LLM-as-judge. Binary CORRECT/WRONG scoring with generous matching and temporal tolerance. Comparable to Mem0 and other published systems.
- **MCQ mode** (`--mcq`): 10-choice multiple-choice from Percena/locomo-mc10. Deterministic — no LLM judge variance. Non-standard; results not comparable to published numbers.

## Output

The script prints a methodology header, per-category breakdown, recall@K metrics, and latency percentiles. Results can be saved to JSON with `--output path.json`.

## Comparison Context

| System | Reported Accuracy | Protocol | Notes |
|--------|-------------------|----------|-------|
| Mem0 | 66.9% | Free-text + LLM judge | Reported on their GitHub |
| OpenAI Memory | 52.9% | Free-text + LLM judge | Reported |
| Hindsight | 89.6% | Free-text + LLM judge | Research system |
| MnemeFusion | TBD | Free-text + LLM judge | Standard protocol |
| MnemeFusion (MCQ) | 70.2% | MCQ (non-standard) | Internal metric |

*Note: Competitor numbers are unverified reference points. Direct comparison requires running on the same dataset scope with the same protocol.*

## References

- LoCoMo dataset: [snap-research/locomo](https://github.com/snap-research/locomo)
- MCQ variant: [Percena/locomo-mc10](https://huggingface.co/datasets/Percena/locomo-mc10)
