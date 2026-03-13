# Contributing to MnemeFusion

Contributions are welcome! This guide covers how to build, test, and submit changes.

## Prerequisites

- **Rust 1.75+** (`rustup default stable`)
- **Python 3.8+** (for Python bindings)
- **maturin** (`pip install maturin`)

Optional:
- **CUDA toolkit** (for GPU-accelerated entity extraction)
- **A GGUF model** (for entity extraction tests — see README for download instructions)

## Building

```bash
# Core library (Rust)
cargo build -p mnemefusion-core

# Python bindings (CPU-only)
cd mnemefusion-python
maturin develop --release

# Python bindings with entity extraction
maturin develop --release --features entity-extraction

# With CUDA GPU support
maturin develop --release --features entity-extraction-cuda
```

## Testing

```bash
# Run all library tests (510+ tests, no GPU required)
cargo test -p mnemefusion-core --lib

# Run with output
cargo test -p mnemefusion-core --lib -- --nocapture

# Run a specific test module
cargo test -p mnemefusion-core profile

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy -p mnemefusion-core --all-targets -- -D warnings
```

All PRs must pass `cargo test`, `cargo fmt --check`, and `cargo clippy` with zero warnings.

## Code Style

- Standard Rust formatting (`rustfmt` defaults, no custom config)
- Use `thiserror` for error types
- Feature-gate optional dependencies with `#[cfg(feature = "...")]`
- Public APIs should have doc comments with examples
- Tests go in `#[cfg(test)] mod tests` blocks within each module

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with tests
3. Ensure CI passes: `cargo fmt --check && cargo clippy -- -D warnings && cargo test -p mnemefusion-core --lib`
4. Open a PR against `main` with a clear description of the change
5. Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`, `chore:`

## Project Structure

```
mnemefusion-core/src/
├── lib.rs              # Public API exports
├── memory.rs           # MemoryEngine (main entry point)
├── config.rs           # Configuration types
├── error.rs            # Error types
├── types/              # Core data types (Memory, Entity, Profile, Timestamp)
├── storage/            # redb storage engine
├── index/              # Vector (HNSW), BM25, Temporal indexes
├── graph/              # Causal and entity graphs
├── query/              # Query planner, fusion, reranking, intent classification
├── ingest/             # Ingestion pipeline, entity profile updates
├── extraction/         # LLM entity extraction (feature-gated)
└── inference/          # llama-cpp inference engine (feature-gated)
```

## Reporting Issues

Please include:
- MnemeFusion version (or commit SHA)
- OS and Rust version (`rustc --version`)
- Minimal reproduction steps
- Full error output

## License

By contributing, you agree that your contributions will be licensed under the same dual MIT/Apache-2.0 license as the project.
