# MnemeFusion Benchmark Evaluations

This directory contains benchmark evaluation scripts for validating MnemeFusion's retrieval quality against industry-standard datasets.

## Benchmarks

### HotpotQA
Multi-hop question answering benchmark with diverse, explainable questions requiring reasoning over multiple supporting documents.

- **Dataset**: [HotpotQA](https://hotpotqa.github.io/)
- **Samples**: 1,000 validation samples (Phase 2)
- **Metric**: Recall@10 (target: >60%, competitive with DPR baseline)

### LoCoMo
Long-term conversational memory retrieval benchmark for testing multi-session dialogue understanding.

- **Dataset**: [LoCoMo](https://github.com/snap-research/locomo) (Long-term Conversational Memory)
- **Conversations**: 10 conversations (300-600 turns each, ~9K tokens, up to 35 sessions)
- **QA Pairs**: Multiple questions per conversation covering single-hop, multi-hop, temporal reasoning
- **Metric**: Recall@10 (target: >70%)

## Setup

### Prerequisites

1. **Python 3.8+** with pip
2. **GPU** (optional but recommended - tested on 4-6GB VRAM)
3. **MnemeFusion Python bindings** installed

### Install Dependencies

```bash
# From project root
cd tests/benchmarks

# Install Python dependencies
pip install -r requirements.txt

# Install MnemeFusion Python package (if not already installed)
cd ../../mnemefusion-python
maturin develop --release
cd ../../tests/benchmarks
```

## Running Benchmarks

### HotpotQA Evaluation

#### Phase 1: Pipeline Validation (10 samples)

Quick validation to ensure the pipeline works correctly:

```bash
python hotpotqa_eval.py --samples 10
```

**Purpose**: Validate end-to-end pipeline before running full evaluation
**Time**: ~1-2 minutes with GPU
**Output**: `hotpotqa_phase1_results.json`

#### Phase 2: Full Evaluation (1,000 samples)

Full benchmark evaluation:

```bash
python hotpotqa_eval.py --samples 1000
```

**Purpose**: Official benchmark results for Sprint 15 completion
**Time**: ~15-30 minutes with GPU
**Output**: `hotpotqa_phase2_results.json`

#### Options

```bash
# Custom number of samples
python hotpotqa_eval.py --samples 100

# Custom top-k retrieval
python hotpotqa_eval.py --samples 1000 --top-k 20

# Disable GPU (use CPU)
python hotpotqa_eval.py --samples 10 --no-gpu

# Custom output file
python hotpotqa_eval.py --samples 1000 --output my_results.json
```

### LoCoMo Evaluation

#### Phase 1: Pipeline Validation (1 conversation)

Quick validation to ensure the pipeline works correctly:

```bash
python locomo_eval.py --samples 1
```

**Purpose**: Validate end-to-end pipeline before running full evaluation
**Time**: ~1-2 minutes with GPU
**Output**: `locomo_phase1_results.json`

#### Phase 2: Full Evaluation (10 conversations)

Full benchmark evaluation:

```bash
python locomo_eval.py --samples 10
```

**Purpose**: Official benchmark results for Sprint 15 completion
**Time**: ~5-10 minutes with GPU
**Output**: `locomo_phase2_results.json`

#### Options

```bash
# Custom dataset path
python locomo_eval.py --samples 10 --dataset path/to/locomo10.json

# Custom top-k retrieval
python locomo_eval.py --samples 10 --top-k 20

# Disable GPU (use CPU)
python locomo_eval.py --samples 1 --no-gpu

# Custom output file
python locomo_eval.py --samples 10 --output my_results.json
```

**Note**: You need to download the LoCoMo dataset first:
```bash
# Clone the repository
git clone https://github.com/snap-research/locomo.git

# Copy dataset to fixtures
cp locomo/data/locomo10.json tests/benchmarks/fixtures/
```

## Embedding Model

All benchmarks use **bge-base-en-v1.5** by default:
- **Model**: `BAAI/bge-base-en-v1.5`
- **Dimensions**: 768
- **Size**: ~420MB
- **GPU Memory**: ~1-2GB (batch size 32-64)
- **Quality**: State-of-the-art for retrieval (MTEB benchmark)

This model represents modern production usage and provides realistic performance metrics.

## Understanding Results

### Recall@K
**Definition**: Percentage of relevant documents retrieved in top-K results.

**Interpretation**:
- Recall@10 = 0.65 means 65% of supporting documents were found in top 10 results
- Higher is better
- Target: >60% (competitive with DPR baseline)

### MRR (Mean Reciprocal Rank)
**Definition**: Average of 1/rank for first relevant document.

**Interpretation**:
- MRR = 0.5 means first relevant document is at rank 2 on average
- Range: 0.0 to 1.0
- Higher is better

### Precision@K
**Definition**: Percentage of retrieved documents that are relevant.

**Interpretation**:
- Precision@10 = 0.3 means 30% of top 10 results were relevant
- Trade-off with recall
- Higher is better

## Expected Performance

Based on Sprint 15 acceptance criteria:

| Benchmark | Metric | Target | Status |
|-----------|--------|--------|--------|
| HotpotQA | Recall@10 | >60% | 🚧 Phase 2 Running |
| LoCoMo | Recall@10 | >70% | ⏳ Pending |

## Troubleshooting

### GPU Not Detected

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Error

Reduce batch size or use CPU:

```bash
python hotpotqa_eval.py --samples 10 --no-gpu
```

### Import Error: mnemefusion

Ensure Python bindings are installed:

```bash
cd ../../mnemefusion-python
maturin develop --release
```

### Dataset Download Fails

Check internet connection and Hugging Face datasets:

```bash
pip install --upgrade datasets
```

## File Structure

```
tests/benchmarks/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── hotpotqa_eval.py          # HotpotQA evaluation script
├── locomo_eval.py            # LoCoMo evaluation script (TODO)
├── fixtures/
│   ├── hotpotqa_phase1_results.json
│   └── hotpotqa_phase2_results.json
└── common/                   # Shared utilities (future)
```

## Sprint 15 Completion

Sprint 15 Week 2 requires:
- ✅ Phase 1: Pipeline validation (10 samples) - Verify it works
- ⏳ Phase 2: Full evaluation (1,000 samples) - Official benchmark

Both phases must complete successfully for Sprint 15 to be marked COMPLETE.
