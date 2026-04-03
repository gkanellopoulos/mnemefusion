# Changelog

All notable changes to MnemeFusion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-04-03

### Added
- Pipeline trace recorder: config-gated step-by-step query diagnostics with 20+ instrumented
  pipeline steps (`enable_trace` config, `last_query_trace()` API)
- Entity-dampened RRF fusion (0.3 scale) to reduce entity flooding in score fusion
- Pre-fusion semantic quality gate with entity exemption
- Dialog bridging: adjacent turn injection for conversational context
- Aggregation candidate pool scaling with query limit
- LongMemEval per-entity evaluation mode (`--mode per-entity`)
- Clean benchmark scripts for LoCoMo and LongMemEval (fresh ingestion, no post-processing)

### Changed
- MMR diversity parameter (λ) raised from 0.7 to 0.9 to preserve same-topic evidence
- Wider semantic fetch with entity-proportional narrowing
- README rewritten around atomic architecture (one database per entity)
- LongMemEval `--mode atomic` renamed to `--mode per-entity`

### Updated
- petgraph 0.6 → 0.8
- GitHub Actions: upload-artifact v7, download-artifact v8

## [0.1.3] - 2026-03-17

### Fixed
- GPU backend discovery: `__init__.py` auto-sets `MNEMEFUSION_DLL_DIR` so `ggml-cuda.so` and
  `ggml-cpu.so` are found in the pip wheel layout without manual env vars

## [0.1.2] - 2026-03-16

### Added
- GPU wheel build in release workflow (Linux x86_64, CUDA 12.8, sm_75 through sm_90)
- GPU wheel published to PyPI as `mnemefusion` (production package)

### Fixed
- `set_embedding_fn()` now works with `add()` and `query()` — previously the callback was
  stored but never consulted when auto-computing embeddings (#31)
- README and `examples/minimal.py` updated with correct `embedding_dim` and prerequisites
- GPU wheel RPATH and soname symlinks for bundled backend shared libraries

## [0.1.1] - 2026-03-15

### Added
- README displayed on crates.io package page
- crates.io, PyPI, and docs.rs badges in README
- docs.rs metadata for auto-generated API documentation

### Changed
- Python install: `pip install mnemefusion-cpu` (was build-from-source)
- Rust install: `mnemefusion-core = "0.1"` (was git dependency)
- Replaced `[patch.crates-io]` git overrides with published fork crates
  (`mnemefusion-llama-cpp-2` and `mnemefusion-llama-cpp-sys-2` on crates.io)

## [0.1.0] - 2026-03-14

### Added
- Initial release to PyPI (`mnemefusion-cpu`) and crates.io (`mnemefusion-core`)
- Pre-built CPU wheels for Linux x86_64, macOS x86_64, macOS arm64, Windows x86_64

## [0.1.0-alpha] - 2026-03-13

Initial public release.

### Added

**Core Engine**
- Single-file `.mfdb` database format with ACID transactions (redb)
- HNSW vector index for semantic similarity search (usearch)
- BM25 keyword index with Porter stemming for exact term matching
- Temporal index with time-range queries and recency scoring
- Causal graph for multi-hop cause-effect relationship traversal (petgraph)
- Entity graph with automatic entity-memory linking
- Reciprocal Rank Fusion (RRF) across all five retrieval dimensions
- MMR diversity reranking to reduce result redundancy
- Intent classification for automatic query routing
- Namespace isolation for multi-user memory separation
- Metadata filtering with AND-logic key-value predicates
- Batch operations: `add_batch`, `delete_batch`, `add_with_dedup`, `upsert`
- Checkpoint/resume system for crash-safe bulk ingestion

**Entity Extraction**
- LLM-powered entity extraction via llama-cpp (GGUF models)
- Multi-pass diverse extraction with configurable perspectives
- Entity profile system with structured facts, summaries, and source linking
- Alias resolution (e.g., "Mel" → "Melanie") at both query and ingestion time
- Profile consolidation and garbage filtering
- First-person pronoun resolution (`set_user_entity`) for conversational memory
- Speaker-aware embedding with pronoun substitution

**Python Bindings**
- Full PyO3 bindings covering all core operations
- `Memory` class with context manager support
- `set_embedding_fn()` for automatic text vectorization
- `query()` returning `(intent, results, profile_context)` tuple
- LLM extraction via `enable_llm_entity_extraction()` or config
- Async extraction queue with `flush_extraction_queue()`

**GPU Support**
- Optional CUDA acceleration for entity extraction
- Automatic GPU layer offloading with configurable layer count
- GPU context auto-reset for long-running ingestion jobs

[0.1.3]: https://github.com/gkanellopoulos/mnemefusion/releases/tag/v0.1.3
[0.1.2]: https://github.com/gkanellopoulos/mnemefusion/releases/tag/v0.1.2
[0.1.1]: https://github.com/gkanellopoulos/mnemefusion/releases/tag/v0.1.1
[0.1.0]: https://github.com/gkanellopoulos/mnemefusion/releases/tag/v0.1.0a1
[0.1.0-alpha]: https://github.com/gkanellopoulos/mnemefusion/releases/tag/v0.1.0-alpha
