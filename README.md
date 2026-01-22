# MnemeFusion

**Unified memory engine for AI applications—"SQLite for AI memory."**

MnemeFusion provides four-dimensional memory indexing (semantic, temporal, causal, entity) in a single embedded database file with zero external dependencies.

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Status: 🎉 Phase 1 COMPLETE ✅

**All 8 sprints of Phase 1 finished!** (January 21, 2026)

Phase 1 achievements:
- ✅ **Storage Layer**: redb-based single-file database with ACID guarantees
- ✅ **Vector Search**: usearch HNSW index for semantic similarity
- ✅ **Temporal Indexing**: Time-based range queries and recency search
- ✅ **Causal Graph**: Multi-hop causal relationship traversal
- ✅ **Entity Graph**: Automatic entity extraction and entity-memory linking
- ✅ **Ingestion Pipeline**: Atomic operations across all dimensions
- ✅ **Query Intelligence**: Intent classification and adaptive fusion
- ✅ **Python Bindings**: Production-ready PyO3 bindings with comprehensive tests

**Test Status:**
- 166 Rust tests passing (133 unit + 12 integration + 21 doc)
- 21 Python tests passing
- All features validated and working

**Next:** Phase 2 will add essential features (provenance, batch operations, deduplication, namespaces, metadata filtering) based on competitive analysis.

## Features

- **Single File Storage**: All data in one portable `.mfdb` file
- **ACID Transactions**: Built on redb for reliability
- **Four Dimensions**: Semantic, temporal, causal, and entity indexing ✅
- **Intent Classification**: Automatic query type detection (temporal, causal, entity, factual)
- **Adaptive Fusion**: Smart weighting of dimensions based on intent
- **Zero Dependencies**: Embedded library, no servers to deploy
- **Rust Core**: Memory-safe, high-performance implementation
- **Python Bindings**: First-class Python API with PyO3 ✅

## Quick Start

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
mnemefusion-core = "0.1"
```

Basic usage:

```rust
use mnemefusion_core::{MemoryEngine, Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open or create a database
    let engine = MemoryEngine::open("./brain.mfdb", Config::default())?;

    // Add a memory (with your embedding vector)
    let embedding = vec![0.1; 384]; // Your embedding model output
    let id = engine.add(
        "Project deadline moved to March 15th".to_string(),
        embedding,
        None, // Optional metadata
        None, // Optional timestamp
    )?;

    // Retrieve a memory
    if let Some(memory) = engine.get(&id)? {
        println!("Content: {}", memory.content);
    }

    // Close when done
    engine.close()?;
    Ok(())
}
```

### Python ✅

```python
import mnemefusion

# Open database
memory = mnemefusion.Memory("./brain.mfdb")

# Add memory with metadata
memory_id = memory.add(
    content="Meeting cancelled due to storm",
    embedding=[0.1] * 384,  # Your embedding model output
    metadata={"type": "event", "priority": "high"}
)

# Intelligent query with intent classification
intent, results = memory.query(
    query_text="Why was the meeting cancelled?",
    query_embedding=[0.1] * 384,
    limit=10
)

print(f"Intent: {intent['intent']} (confidence: {intent['confidence']})")

for mem, scores in results:
    print(f"Fused score: {scores['fused_score']}")
    print(f"Content: {mem['content']}")

# Or use context manager (recommended)
with mnemefusion.Memory("./brain.mfdb") as memory:
    memory.add("Some content", embedding)
    # Automatically closes on exit
```

**Installation (Development):**
```bash
cd mnemefusion-python
python -m venv .venv
.venv/Scripts/activate  # On Windows
pip install maturin
maturin develop
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for a comprehensive tutorial.

## Architecture

```
┌─────────────────────────────────────────┐
│         MemoryEngine (Public API)        │
├─────────────────────────────────────────┤
│  Storage Layer (redb)                   │
│  - ACID transactions                    │
│  - Single file format                   │
│  - Memory CRUD                          │
├─────────────────────────────────────────┤
│  Index Layer (Coming Sprint 2-3)       │
│  - Vector Index (usearch/HNSW)          │
│  - Temporal Index (B-tree)              │
├─────────────────────────────────────────┤
│  Graph Layer (Coming Sprint 4-5)        │
│  - Causal Graph (petgraph)              │
│  - Entity Graph (petgraph)              │
├─────────────────────────────────────────┤
│  Query Layer (Coming Sprint 7-8)        │
│  - Intent Classification                │
│  - Adaptive Fusion                      │
└─────────────────────────────────────────┘
```

## Building from Source

### Prerequisites

- Rust 1.75 or later
- (Optional) Python 3.8+ for Python bindings

### Build

```bash
# Clone repository
git clone https://github.com/yourusername/mnemefusion.git
cd mnemefusion

# Build core library
cargo build --release

# Run tests
cargo test --all

# Run example
cargo run --example basic_usage
```

## Development Roadmap

### Phase 1: Core Engine (4 months)

| Sprint | Status | Focus |
|--------|--------|-------|
| Sprint 1 | ✅ Complete | Foundation, storage, CRUD |
| Sprint 2 | 🔄 Next | Vector index (usearch), semantic search |
| Sprint 3 | ⏳ Planned | Temporal indexing |
| Sprint 4 | ⏳ Planned | Causal graph |
| Sprint 5 | ⏳ Planned | Entity graph |
| Sprint 6 | ⏳ Planned | Ingestion pipeline |
| Sprint 7 | ⏳ Planned | Query planner, intent classification |
| Sprint 8 | ⏳ Planned | Fusion engine, Python bindings |

### Phase 2: Production Hardening (3 months)

- ACID guarantees & crash recovery
- Performance optimization (<10ms search latency)
- Comprehensive testing (>80% coverage)
- API stability & documentation
- PyPI distribution

### Phase 3: Ecosystem

- Community building
- Enterprise features
- Additional language bindings
- Advanced entity extraction

## Testing

### Rust Tests
```bash
# Run all Rust tests
cargo test --all

# Run core tests only
cargo test --package mnemefusion-core

# Run with output
cargo test -- --nocapture
```

### Python Tests
```bash
cd mnemefusion-python
.venv/Scripts/activate
pytest tests/ -v
```

**Current test coverage:**
- ✅ **133 Rust unit tests** across all core modules
- ✅ **12 integration tests** covering end-to-end workflows
- ✅ **21 doc tests** ensuring examples compile
- ✅ **21 Python tests** validating bindings
- **Total: 187 tests, all passing**

See [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md) for detailed test results.

## Configuration

```rust
use mnemefusion_core::Config;

let config = Config::new()
    .with_embedding_dim(512)          // Default: 384
    .with_temporal_decay_hours(336.0) // Default: 168 (1 week)
    .with_causal_max_hops(5)          // Default: 3
    .with_entity_extraction(true);    // Default: true

let engine = MemoryEngine::open("./brain.mfdb", config)?;
```

## File Format

MnemeFusion uses a custom `.mfdb` (MnemeFusion Database) format:

```
┌─────────────────────────────┐
│  Header (64 bytes)          │
│  - Magic: "MFDB"            │
│  - Version: 1               │
│  - Timestamps               │
├─────────────────────────────┤
│  redb Tables                │
│  - memories                 │
│  - temporal_index           │
│  - metadata                 │
│  - (more in future sprints) │
└─────────────────────────────┘
```

Version 1 guarantees:
- Forward compatibility within major version
- File format stability
- Safe concurrent reads
- ACID write transactions

## Performance Targets

| Operation | Target | Status |
|-----------|--------|--------|
| Add memory | < 10ms | ✅ Achieved (~1ms) |
| Get by ID | < 1ms | ✅ Achieved (~0.1ms) |
| Search (100K) | < 10ms | Sprint 2 |
| Search (1M) | < 50ms | Sprint 10 |

## Contributing

MnemeFusion is currently in active development. Contributions will be welcome after Sprint 14 (1.0 release candidate).

For now, feel free to:
- Report issues
- Suggest features
- Star the repository
- Follow development progress

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Documentation

- **[Getting Started Guide](GETTING_STARTED.md)** - Step-by-step tutorial for new users
- **[Python API Reference](mnemefusion-python/README.md)** - Complete Python API documentation
- **[Validation Results](VALIDATION_RESULTS.md)** - Phase 1 testing and validation report
- **[Developer Guide](CLAUDE.md)** - For contributors
- **[Project State](PROJECT_STATE.md)** - Current status and sprint history
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Full roadmap (Phases 1-4)
- **[Feature Roadmap](mnemefusion_feature_roadmap.md)** - Competitive analysis and future features

## Links

- **Examples**: [Rust examples](./mnemefusion-core/examples/) | [Python examples](./mnemefusion-python/examples/)
- **Issue Tracker**: [GitHub Issues](https://github.com/yourusername/mnemefusion/issues)
- **Project Plan**: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)

## Acknowledgments

Built on excellent Rust libraries:
- [redb](https://github.com/cberner/redb) - Embedded database
- [usearch](https://github.com/unum-cloud/usearch) - Vector search (Sprint 2)
- [petgraph](https://github.com/petgraph/petgraph) - Graph algorithms (Sprint 4-5)

## Status Updates

**January 21, 2026** - 🎉 **Phase 1 COMPLETE!** All 8 sprints finished. 187 tests passing. Python bindings fully functional. Ready for Phase 2 (essential features).

**January 14, 2026** - Sprint 1 complete! Core foundation solid with 63 passing tests.

---

**"SQLite for AI memory"** - One file. Four dimensions. Zero complexity.
