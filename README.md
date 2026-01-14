# MnemeFusion

**Unified memory engine for AI applications—"SQLite for AI memory."**

MnemeFusion provides four-dimensional memory indexing (semantic, temporal, causal, entity) in a single embedded database file with zero external dependencies.

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Status: Phase 1 - Sprint 1 Complete ✅

Currently in active development. Sprint 1 objectives achieved:
- ✅ Core Rust project structure
- ✅ Storage layer with redb
- ✅ Memory CRUD operations
- ✅ File format with validation
- ✅ 50+ unit tests passing
- ✅ 6 integration tests passing
- ✅ Working example code

**Next:** Sprint 2 will add vector indexing with usearch for semantic search.

## Features

- **Single File Storage**: All data in one portable `.mfdb` file
- **ACID Transactions**: Built on redb for reliability
- **Four Dimensions**: Semantic, temporal, causal, and entity indexing (in progress)
- **Zero Dependencies**: Embedded library, no servers to deploy
- **Rust Core**: Memory-safe, high-performance implementation
- **Python Bindings**: First-class Python API (coming Sprint 8)

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

### Python (Coming in Sprint 8)

```python
from mnemefusion import Memory, Config

# Open database
memory = Memory("./brain.mfdb")

# Add memory
memory_id = memory.add("Meeting cancelled due to storm", embedding)

# Search (intent-aware)
results = memory.search("Why was the meeting cancelled?", query_embedding)

memory.close()
```

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

```bash
# Run all tests
cargo test --all

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_memory_engine

# Run integration tests
cargo test --test integration_test
```

Current test coverage:
- **50 unit tests** across all core modules
- **6 integration tests** covering end-to-end workflows
- **7 doc tests** ensuring examples compile

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

## Links

- **Documentation**: Coming in Sprint 12
- **Examples**: [examples/](./mnemefusion-core/examples/)
- **Issue Tracker**: [GitHub Issues](https://github.com/yourusername/mnemefusion/issues)
- **Project Plan**: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)

## Acknowledgments

Built on excellent Rust libraries:
- [redb](https://github.com/cberner/redb) - Embedded database
- [usearch](https://github.com/unum-cloud/usearch) - Vector search (Sprint 2)
- [petgraph](https://github.com/petgraph/petgraph) - Graph algorithms (Sprint 4-5)

## Status Updates

**January 14, 2026** - Sprint 1 complete! Core foundation solid with 56 passing tests. Moving to Sprint 2 for vector indexing.

---

**"SQLite for AI memory"** - One file. Four dimensions. Zero complexity.
