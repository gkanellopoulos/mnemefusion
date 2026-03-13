# Changelog

All notable changes to MnemeFusion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0-alpha]: https://github.com/gkanellopoulos/mnemefusion/releases/tag/v0.1.0-alpha
