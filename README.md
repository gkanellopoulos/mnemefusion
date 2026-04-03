# MnemeFusion

**Atomic memory engine for AI applications — one database per entity.**

MnemeFusion gives each entity its own self-contained memory database. Five retrieval dimensions (semantic, keyword, temporal, causal, entity profile) are fused into a single ranked result, all in one portable `.mfdb` file with zero external dependencies.

Think SQLite for AI memory: one file per user, per contact, or per conversation — embedded in your application.

[![CI](https://github.com/gkanellopoulos/mnemefusion/actions/workflows/ci.yml/badge.svg)](https://github.com/gkanellopoulos/mnemefusion/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnemefusion-core.svg)](https://crates.io/crates/mnemefusion-core)
[![PyPI CPU](https://img.shields.io/pypi/v/mnemefusion-cpu.svg?label=pypi%20cpu)](https://pypi.org/project/mnemefusion-cpu/)
[![PyPI GPU](https://img.shields.io/pypi/v/mnemefusion.svg?label=pypi%20gpu)](https://pypi.org/project/mnemefusion/)
[![docs.rs](https://docs.rs/mnemefusion-core/badge.svg)](https://docs.rs/mnemefusion-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

*MnemeFusion was designed and directed by [George Kanellopoulos](https://github.com/gkanellopoulos), with implementation substantially assisted by [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Anthropic). The project grew out of an exploration into building a complex, multi-dimensional AI memory engine through human-AI collaboration — the commit history reflects the authentic development process.*

## Atomic Architecture

MnemeFusion follows an **atomic design**: each entity (a user, a contact, a conversation) maps to its own `.mfdb` database file. This 1:1 mapping is the core architectural principle.

Memory retrieval degrades when unrelated conversations share a database — relevant memories get buried by noise from other entities. By scoping each database to a single entity, all five retrieval dimensions stay focused and retrieval stays precise, even as conversation history grows to thousands of turns.

This mirrors how production AI systems work: a personal assistant remembers *one user's* conversations, a CRM agent tracks *one contact's* history, a therapy bot maintains *one patient's* sessions. Each gets its own `.mfdb` file.

## Features

- **Five Retrieval Pathways**: Semantic vector search, BM25 keyword matching, temporal range queries, causal graph traversal, entity profile scoring
- **Reciprocal Rank Fusion**: Fuses all five dimensions into a single ranked result set
- **Entity Profiles**: LLM-powered entity extraction builds structured knowledge graphs from unstructured text
- **Single File Storage**: All data in one portable `.mfdb` file with ACID transactions (redb)
- **Intent Classification**: Automatic query routing (temporal, causal, entity, factual)
- **Namespace Isolation**: Multi-user memory separation
- **Rust Core**: Memory-safe, high-performance embedded library
- **Python Bindings**: First-class Python API via PyO3
- **Optional GPU Acceleration**: CUDA-accelerated entity extraction via llama-cpp

## Benchmarks

Evaluated on two established conversational memory benchmarks using standard protocols.

**[LoCoMo](evals/locomo/)** — 1,540 free-text questions across 10 multi-session conversations:

| Accuracy |
|----------|
| **69.9% ± 0.4%** |

**[LongMemEval](evals/longmemeval/)** — 500 binary questions, three evaluation modes:

| Mode | What it tests | Score |
|------|---------------|-------|
| Oracle | Pipeline quality — extraction + RAG + scoring | **91.4%** |
| Per-entity | Production pattern — one DB per conversation | **67.6%** |
| Shared DB | All conversations in one DB — the anti-pattern | 37.2% |

The oracle result (91.4%) proves the pipeline works when given the right evidence. The gap to per-entity (67.6%) is a retrieval problem: 48% of failures had zero gold evidence in the top-20 results, 49% had partial evidence, and only 2.5% were reasoning failures. The shared-DB collapse (37.2%) shows why per-entity scoping matters — unrelated conversations compete for retrieval slots.

### Competitive context

Every system below requires cloud LLM APIs for its memory engine. MnemeFusion is the only one that runs entirely locally — entity extraction uses a 3.8B GGUF model on-device, and embeddings are computed locally. Only the answer generation step uses a cloud LLM, and that is the application's choice, not a library dependency.

| System | Answer model | LongMemEval-S | LoCoMo | Memory engine runs locally |
|--------|-------------|---------------|--------|---------------------------|
| [Mnemis](https://github.com/microsoft/Mnemis) (Microsoft) | GPT-4.1-mini | 91.6% | 93.9% | No — Azure OpenAI + Neo4j |
| [Hindsight](https://github.com/vectorize-io/hindsight) (Vectorize) | GPT-OSS-120B | 89.0% | 85.7% | No — cloud LLM via Groq |
| [EverMemOS](https://github.com/EverMind-AI/EverMemOS) | GPT-4.1-mini | 83.0% | 92.3% | No — MongoDB + ES + Milvus + cloud LLM |
| [MemMachine](https://github.com/MemMachine/MemMachine) | GPT-4.1-mini | — | 91.2% | No — Neo4j + OpenAI + Cohere |
| [Zep](https://github.com/getzep/graphiti) | GPT-4o | 71.2% | 75.1%\* | No — Zep Cloud + OpenAI |
| [Mem0](https://github.com/mem0ai/mem0) | GPT-4o-mini | — | 66.9%\*\* | No — platform + OpenAI |
| **MnemeFusion** | **Phi-4-mini 3.8B (local)** | **67.6% / 91.4% oracle** | **69.9%** | **Yes** |

\*Zep's LoCoMo score is [disputed](https://github.com/getzep/zep-papers/issues/5) — independently measured at 58.4% by Mem0.
\*\*Mem0's published score uses [platform-only features](https://github.com/mem0ai/mem0/issues/2800); the open-source version scores ~34%.

The accuracy gap is largely driven by the answer model. Systems using GPT-4.1-mini or GPT-OSS-120B gain 15-25 points from model intelligence alone — the same retrieval results produce higher scores with a stronger answer model. MnemeFusion's oracle result (91.4%) demonstrates that retrieval quality is competitive when the evidence is present; the bottleneck is extraction coverage with a local 3.8B model, not the retrieval architecture.

All competitor numbers are from their published papers or repositories. Sources: Mnemis ([arXiv 2602.15313](https://arxiv.org/abs/2602.15313)), Hindsight ([arXiv 2512.12818](https://arxiv.org/abs/2512.12818)), EverMemOS ([arXiv 2601.02163](https://arxiv.org/abs/2601.02163)), Zep ([arXiv 2501.13956](https://arxiv.org/abs/2501.13956)), Mem0 ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413)), MemMachine ([blog](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/)).

LongMemEval uses the official binary protocol with gpt-4o-2024-08-06 as judge. LoCoMo uses free-text generation with LLM-as-judge (GPT-4o-mini). See [evals/](evals/) for full methodology, datasets, and reproduction instructions.

## Quick Start

For a complete runnable example, see [`examples/minimal.py`](examples/minimal.py) — no GPU or GGUF model required. For an interactive demo, see the [Chat Demo](apps/) (Streamlit).

### Python

```bash
# CPU-only (development / experimentation)
pip install mnemefusion-cpu sentence-transformers

# GPU with CUDA (production — Linux x86_64, requires NVIDIA driver 525+)
pip install mnemefusion sentence-transformers
```

```python
import mnemefusion
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Open or create a database (768 = BGE-base embedding dimension)
mem = mnemefusion.Memory("./brain.mfdb", {"embedding_dim": 768})

# Set embedding function for automatic vectorization
mem.set_embedding_fn(lambda text: model.encode(text).tolist())

# Add memories
mem.add("Alice loves hiking in the mountains", metadata={"speaker": "narrator"})
mem.add("Bob started learning piano last month", metadata={"speaker": "narrator"})

# Multi-dimensional query — returns (intent, results, profile_context)
intent, results, profiles = mem.query("What are Alice's hobbies?", limit=10)

print(f"Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
for memory_dict, scores_dict in results:
    print(f"  [{scores_dict['fused_score']:.3f}] {memory_dict['content']}")

# Profile context contains entity facts for RAG augmentation
for fact_str in profiles:
    print(f"  Profile: {fact_str}")
```

### With User Identity

```python
# Namespace isolation + first-person pronoun resolution
mem = mnemefusion.Memory("./brain.mfdb", {"embedding_dim": 768}, user="alice")
mem.set_embedding_fn(lambda text: model.encode(text).tolist())

# Memories are namespaced to "alice"
mem.add("I love hiking in the mountains")

# Map "I"/"me"/"my" → "alice" entity profile at query time
mem.set_user_entity("alice")

# "my hobbies" resolves to alice's profile
intent, results, profiles = mem.query("What are my hobbies?")
```

### With LLM Entity Extraction

Entity extraction uses a local GGUF model (no cloud API needed). Download a supported model:

```bash
pip install huggingface-hub

# Recommended: Phi-4-mini (3.8B, ~2.3GB, best accuracy)*
# Requires Hugging Face authentication: huggingface-cli login
huggingface-cli download microsoft/Phi-4-mini-instruct-gguf Phi-4-mini-instruct-Q4_K_M.gguf --local-dir models/

# Alternative (no auth required): Qwen2.5-3B (~2GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir models/
```

*\*MnemeFusion's extraction prompts have been tested and tuned with Phi-4-mini. Other models may work but with reduced extraction quality.*

```python
mem = mnemefusion.Memory("./brain.mfdb", {"embedding_dim": 768})
mem.set_embedding_fn(lambda text: model.encode(text).tolist())
mem.enable_llm_entity_extraction("models/Phi-4-mini-instruct-Q4_K_M.gguf", tier="balanced")

# Entity extraction runs automatically on add()
mem.add("Caroline studies marine biology at Stanford")

# Entity profiles are built incrementally
profile = mem.get_entity_profile("caroline")
# {'name': 'caroline', 'entity_type': 'person', 'facts': {...}, 'summary': '...'}
```

Requires a GPU with 4GB+ VRAM for reasonable speed. CPU-only works but is ~10x slower. For GPU acceleration, install the GPU package: `pip install mnemefusion`.

### Rust

```toml
[dependencies]
mnemefusion-core = "0.1"
```

```rust
use mnemefusion_core::{MemoryEngine, Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = MemoryEngine::open("./brain.mfdb", Config::default())?;

    // Add a memory with embedding vector
    let embedding = vec![0.1; 384]; // From your embedding model
    engine.add(
        "Project deadline moved to March 15th".to_string(),
        embedding,
        None, // metadata
        None, // timestamp
        None, // source
        None, // namespace
    )?;

    // Query with multi-dimensional fusion
    let query_embedding = vec![0.1; 384];
    let (_intent, results, _profiles) = engine.query(
        "When is the project deadline?",
        query_embedding,
        10,    // limit
        None,  // namespace
        None,  // filters
    )?;

    for (memory, scores) in &results {
        println!("[{:.3}] {}", scores.fused_score, memory.content);
    }

    engine.close()?;
    Ok(())
}
```

## Bring Your Own Model

MnemeFusion has three model slots — all pluggable, none locked to a specific provider:

| Slot | What it does | Default | Alternatives |
|------|-------------|---------|--------------|
| **Embedding model** | Vectorizes memories and queries | Any [sentence-transformers](https://www.sbert.net/) model via `set_embedding_fn()` | OpenAI, Cohere, Voyage, custom fine-tuned — anything that returns a vector |
| **Extraction LLM** | Extracts entities and facts at ingestion time | [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct) 3.8B GGUF (local, optional) | Any instruction-tuned GGUF model — Qwen, Llama, Mistral. Or disable entirely; four of five retrieval dimensions still work |
| **Answer LLM** | Generates the final response from retrieved context | *Not included* — this is your application's choice | Claude, GPT, Gemini, a local model, anything. MnemeFusion returns ranked results; what generates the answer is up to you |

The local-first design is a privacy architecture, not just a cost optimization. Conversational memory often contains sensitive personal data. Running entity extraction on-device means memory content never leaves the machine. Users who want cloud-grade extraction quality can point the embedding slot at a cloud API and still keep extraction local, or swap in a larger local model as hardware allows.

## Architecture

![MnemeFusion Architecture](mnemefusion_architecture_v2.svg)

## Python API Reference

### Core Operations

| Method | Description |
|--------|-------------|
| `Memory(path, config=None, user=None)` | Open or create a database |
| `add(content, embedding=None, metadata=None, timestamp=None, source=None, namespace=None)` | Add a memory |
| `query(query_text, query_embedding=None, limit=10, namespace=None, filters=None)` | Multi-dimensional query returning `(intent, results, profiles)` |
| `search(query_embedding, top_k, namespace=None, filters=None)` | Pure semantic similarity search |
| `get(memory_id)` | Retrieve memory by ID |
| `delete(memory_id)` | Delete memory by ID |
| `close()` | Close database and save indexes |

### Batch Operations

| Method | Description |
|--------|-------------|
| `add_batch(memories, namespace=None)` | Bulk insert (10x+ faster) |
| `add_with_dedup(content, embedding, ...)` | Add with duplicate detection |
| `upsert(key, content, embedding, ...)` | Insert or update by logical key |
| `delete_batch(memory_ids)` | Bulk delete |

### Entity & Profile Management

| Method | Description |
|--------|-------------|
| `enable_llm_entity_extraction(model_path, tier="balanced", extraction_passes=1)` | Enable LLM extraction |
| `set_user_entity(name)` | Map first-person pronouns to user entity |
| `list_entity_profiles()` | List all entity profiles |
| `get_entity_profile(name)` | Get profile by name (case-insensitive) |
| `consolidate_profiles()` | Remove noise from profiles |
| `summarize_profiles()` | Generate profile summaries |

### Diagnostics

| Method | Description |
|--------|-------------|
| `last_query_trace()` | Step-by-step trace of the most recent `query()` call (requires `enable_trace=True` in config) |

### Metadata Filtering

```python
# Filter by metadata key-value pairs (AND logic)
filters = [
    {"metadata_key": "speaker", "metadata_value": "Alice"},
    {"metadata_key": "session", "metadata_value": "2024-01-15"},
]
intent, results, profiles = mem.query("hiking plans", filters=filters)
```

### Namespace System

```python
# Add to specific namespace
mem.add("secret note", namespace="alice")

# Query within namespace
intent, results, profiles = mem.query("notes", namespace="alice")

# Or use the user= constructor shortcut
mem = mnemefusion.Memory("brain.mfdb", user="alice")
# All add/query calls default to the "alice" namespace
```

## Configuration

```python
config = {
    "embedding_dim": 384,              # Must match your embedding model
    "entity_extraction_enabled": True,  # Enable built-in entity extraction
    "llm_model": "path/to/model.gguf", # Auto-enables LLM extraction
    "extraction_passes": 3,             # Multi-pass diverse extraction
    "async_extraction_threshold": 500,  # Defer extraction for large docs
    "enable_trace": True,               # Record step-by-step query traces
}
mem = mnemefusion.Memory("brain.mfdb", config=config)
```

```rust
use mnemefusion_core::Config;

let config = Config::new()
    .with_embedding_dim(384)
    .with_entity_extraction(true);

let engine = MemoryEngine::open("./brain.mfdb", config)?;
```

## Error Handling

All errors surface as standard Python exceptions — no custom exception types.

| Exception | When | Recoverable |
|-----------|------|-------------|
| `IOError` | Database open/close fails, disk full, file not found, concurrent open of same file | Usually yes (fix path, free disk, close other instance) |
| `ValueError` | Wrong embedding dimension, invalid memory ID, bad config | Yes (fix input) |
| `RuntimeError` | Calling methods after `close()` | Reopen with a new `Memory()` instance |

```python
import mnemefusion

mem = mnemefusion.Memory("brain.mfdb")

# After close(), all operations raise RuntimeError
mem.close()
try:
    mem.add("text")
except RuntimeError as e:
    print(e)  # "Database is closed"

# Each .mfdb file supports one open instance at a time
mem1 = mnemefusion.Memory("brain.mfdb")
try:
    mem2 = mnemefusion.Memory("brain.mfdb")  # Same file
except IOError as e:
    print(e)  # File lock error
```

## Building from Source

### Prerequisites

- Rust 1.75+
- Python 3.9+ (for Python bindings)

### Build

```bash
git clone https://github.com/gkanellopoulos/mnemefusion.git
cd mnemefusion

# Build core library
cargo build --release

# Run tests (520+ tests)
cargo test -p mnemefusion-core --lib

# Build Python bindings
cd mnemefusion-python
maturin develop --release

# With CUDA GPU support (requires CUDA toolkit)
maturin develop --release --features entity-extraction-cuda
```

## Testing

```bash
# All library unit tests
cargo test -p mnemefusion-core --lib

# With output
cargo test -p mnemefusion-core --lib -- --nocapture

# Run specific test module
cargo test -p mnemefusion-core profile
```

## Language Support

MnemeFusion's core search works with any language via multilingual embeddings. Entity extraction and intent classification are currently English-optimized.

| Feature | Language Support |
|---------|-----------------|
| Vector search | All languages (use multilingual embeddings) |
| BM25 keyword search | English-optimized (Porter stemming) |
| Temporal indexing | All languages |
| Causal links | All languages |
| Entity extraction | English (optional, can be disabled) |
| Metadata filtering | All languages |

For non-English use, disable entity extraction:

```python
config = {"entity_extraction_enabled": False, "embedding_dim": 768}
mem = mnemefusion.Memory("brain.mfdb", config=config)
```

## API Stability

MnemeFusion is pre-1.0. The following APIs are considered **stable** and will not change without a version bump:

| API | Stable Since |
|-----|-------------|
| `Memory(path, config, user)` | 0.1.0 |
| `add(content, embedding, metadata, timestamp)` | 0.1.0 |
| `query(query_text, query_embedding, limit, namespace, filters)` | 0.1.0 |
| `search(query_embedding, top_k, namespace, filters)` | 0.1.0 |
| `get(memory_id)` / `delete(memory_id)` | 0.1.0 |
| `close()` | 0.1.0 |
| `add_batch(memories, namespace)` | 0.1.0 |
| `set_embedding_fn(fn)` | 0.1.0 |

Everything else (entity extraction API, profile management, config keys) may change between minor versions. The `.mfdb` file format includes embedded version metadata — format-breaking changes will be documented in the [CHANGELOG](CHANGELOG.md).

## Performance Characteristics

| Operation | Complexity | Typical Latency |
|-----------|-----------|-----------------|
| `add()` | O(log n) HNSW insertion + O(n) BM25 update | <5ms without entity extraction |
| `add()` with LLM extraction | Same + LLM inference | ~3-9s depending on GPU |
| `query()` | O(k·log n) across all dimensions + RRF fusion | ~50ms at 5K memories, ~200ms at 50K |
| `search()` | O(k·log n) vector-only | <10ms |
| `get()` / `delete()` | O(1) key lookup | <1ms |
| Storage overhead | ~1.5-2x raw content size (384-dim embeddings) | — |

Tested with up to 10K memories in a single `.mfdb` file. MnemeFusion is designed for per-entity databases — each user, contact, or conversation gets its own `.mfdb` file, typically containing 1K-10K memories. This atomic pattern keeps retrieval precise and scales horizontally.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for build instructions, test commands, and PR guidelines.

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

## Acknowledgments

Built on excellent open-source libraries:
- [redb](https://github.com/cberner/redb) — Embedded key-value store
- [usearch](https://github.com/unum-cloud/usearch) — HNSW vector search
- [petgraph](https://github.com/petgraph/petgraph) — Graph algorithms
- [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) — Rust bindings for llama.cpp
- [PyO3](https://github.com/PyO3/pyo3) — Rust-Python interop
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — AI-assisted development

---

**"SQLite for AI memory"** — One entity, one file. Five dimensions. Zero complexity.
