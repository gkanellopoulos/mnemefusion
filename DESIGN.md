# Design Decisions

This document explains the key architectural choices behind MnemeFusion and the trade-offs considered.

## 1. redb over SQLite

**Choice:** [redb](https://github.com/cberner/redb) — a pure Rust embedded key-value store with ACID transactions.

**Why not SQLite:**
- Pure Rust means no C compiler dependency, no cross-compilation headaches, no `libsqlite3-dev` on CI
- Single-writer with concurrent readers matches our access pattern (ingestion is sequential, queries are parallel)
- B-tree storage with zero-copy reads is well-suited for memory ID lookups
- Embeds cleanly as a library — no process management, no WAL files, just one `.mfdb` file

**Trade-off:** No SQL query language. All queries go through our index layer (vector, BM25, temporal, graph). This is intentional — SQL would add complexity without benefit for our access patterns.

## 2. usearch over FAISS

**Choice:** [usearch](https://github.com/unum-cloud/usearch) — a C++ HNSW vector search library with Rust bindings.

**Why not FAISS:**
- Native Rust bindings (via C FFI) vs. FAISS's Python-first design
- Built-in index persistence (save/load from file) without serialization overhead
- Smaller binary footprint — no OpenBLAS/MKL dependency
- Supports incremental insertion without rebuilding the index

**Trade-off:** Smaller ecosystem and fewer quantization options than FAISS. Acceptable because MnemeFusion targets thousands to tens of thousands of memories per database, not millions — HNSW performance is more than sufficient at this scale.

## 3. Reciprocal Rank Fusion over learned fusion

**Choice:** RRF to combine five retrieval dimensions (semantic, BM25, temporal, entity, causal) into a single ranked result.

**Why not learned fusion:**
- No training data required — works out of the box for any domain
- Deterministic and explainable — each dimension's contribution is traceable
- No model distribution or fine-tuning burden on users
- Research shows RRF performs competitively with learned approaches for heterogeneous retrieval (Cormack et al., 2009)

**Trade-off:** Cannot learn domain-specific dimension weights. In practice, the per-dimension scoring (especially entity profiles and BM25) provides enough signal that fixed fusion works well. The [LoCoMo benchmark](evals/locomo/) validates this at 70.7% accuracy across diverse question types.

## 4. Single file over client-server

**Choice:** Everything in one `.mfdb` file — vectors, keywords, graphs, profiles, metadata.

**Why not a server:**
- Zero deployment complexity — no Docker, no ports, no connection strings
- Portable — copy one file to move an entire memory database
- Embeddable — works inside any application, from CLI tools to mobile apps
- Per-user isolation — one file per user/entity is the natural scaling unit

**Trade-off:** No built-in network access or horizontal scaling. MnemeFusion is an embedded library, not a service. For multi-node deployments, the application layer handles distribution — same model as SQLite.

## 5. Local GGUF models over cloud APIs

**Choice:** Entity extraction runs locally via [llama-cpp](https://github.com/ggerganov/llama.cpp) with quantized GGUF models (Phi-4-mini, Qwen3-4B).

**Why not cloud APIs (OpenAI, Anthropic, etc.):**
- Privacy — conversational memory often contains sensitive personal data
- Cost — entity extraction runs on every `add()` call; cloud API costs compound quickly
- Latency — local inference on GPU is faster than a network round-trip for small models
- Offline capability — works without internet access
- No API key management or rate limiting

**Trade-off:** Requires GPU for reasonable extraction speed (~3-9s per document with GPU, ~30s+ on CPU). Users without a GPU can disable entity extraction and still use the four other retrieval dimensions (semantic, BM25, temporal, causal). The extraction feature is optional — `entity-extraction` is a Cargo feature flag, not a default dependency.
