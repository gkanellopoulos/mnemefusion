# MnemeFusion: Project Brief

## Unified Memory Engine for AI Applications

**Document Version:** 0.1  
**Created:** January 2025  
**Status:** Research & Planning

---

## Vision

A single embedded database engine that natively indexes and queries across four memory dimensions—semantic, temporal, causal, and entity—delivering unified AI memory in one file with zero external dependencies.

**The SQLite of AI memory.**

---

## Problem Statement

Building memory-aware AI applications today requires assembling multiple databases:

| Dimension | Current Solution | Operational Burden |
|-----------|------------------|-------------------|
| Semantic (similarity) | Qdrant, Pinecone, Weaviate | Vector DB deployment |
| Temporal (time) | PostgreSQL, SQLite | Relational DB management |
| Causal (why) | Neo4j, custom | Graph DB expertise |
| Entity (who/what) | Neo4j, custom | Entity resolution logic |

Developers face:
- **Infrastructure complexity**: 3-4 separate databases to deploy and maintain
- **Integration burden**: Custom code to query across systems and fuse results
- **Operational cost**: Each database has its own scaling, backup, monitoring needs
- **Latency overhead**: Network hops between services, result marshaling

**The insight**: These aren't fundamentally different storage problems. They're different indexes over the same underlying data. One engine should handle all four.

---

## Product Vision

### What MnemeFusion Is

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│                                                              │
│    from mnemefusion import Memory                           │
│                                                              │
│    memory = Memory("./brain.mfdb")  # Single file           │
│    memory.add("Meeting cancelled because of storm")          │
│    results = memory.search("Why was meeting cancelled?")     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MnemeFusion                             │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Unified Query Planner                   │   │
│   │     Intent Classification → Adaptive Weights         │   │
│   └─────────────────────────────────────────────────────┘   │
│                              │                               │
│        ┌─────────┬──────────┼──────────┬─────────┐          │
│        ▼         ▼          ▼          ▼         ▼          │
│   ┌────────┐┌────────┐┌──────────┐┌────────┐┌────────┐      │
│   │ Vector ││Temporal││  Causal  ││ Entity ││Payload │      │
│   │ Index  ││ Index  ││  Graph   ││ Graph  ││ Store  │      │
│   │ (HNSW) ││(B-tree)││(Adjacency││(Adjacency│(KV)    │      │
│   └────────┘└────────┘└──────────┘└────────┘└────────┘      │
│                              │                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Unified Storage Layer                   │   │
│   │                  (Single File)                       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│                       brain.mfdb                             │
└─────────────────────────────────────────────────────────────┘
```

### What MnemeFusion Is NOT

- Not a cloud service (embedded only)
- Not a distributed database (single-node)
- Not a framework (pure library)
- Not an LLM wrapper (bring your own models)

---

## Core Thesis

**The components for unified AI memory exist. The integration doesn't.**

| Component | Exists As | Status |
|-----------|-----------|--------|
| HNSW algorithm | Open papers, OSS implementations | ✅ Mature |
| B-tree temporal indexing | Textbook CS, many implementations | ✅ Mature |
| Graph storage | Well-understood data structures | ✅ Mature |
| Query planning | Database engineering discipline | ✅ Mature |
| **Unified 4D memory engine** | Nothing | ❌ Gap |

MnemeFusion fills the integration gap by composing validated algorithms into a single, purpose-built engine for AI memory workloads.

---

## Technical Approach

### Language: Rust

| Reason | Why It Matters |
|--------|----------------|
| Memory safety | Managing indexes, graphs, vectors without GC pauses |
| Performance | SIMD operations, cache-friendly data structures |
| Single binary | No runtime dependencies for end users |
| Python bindings | PyO3 provides seamless Python API |
| Ecosystem | Mature libraries for core algorithms |

### Build Strategy: Compose, Don't Reimplement

We integrate existing battle-tested Rust libraries rather than rewriting algorithms:

| Component | Library | Maturity |
|-----------|---------|----------|
| Vector index (HNSW) | usearch or hora | Production-ready |
| KV storage | redb | Production-ready |
| Graph structures | petgraph | Production-ready |
| Serialization | rkyv or bincode | Production-ready |

**What we build custom:**
- Unified query planner (our core innovation)
- Intent classification
- Adaptive fusion algorithm
- Single-file format coordination
- Graph persistence layer
- Python bindings

### User Experience

```python
# Installation: pip install mnemefusion
# No Rust toolchain needed - pre-built wheels

from mnemefusion import Memory, Config

# Single file, all dimensions included
memory = Memory("./brain.mfdb")

# Or with configuration
memory = Memory(
    "./brain.mfdb",
    config=Config(
        embedding_dim=384,
        temporal_decay_hours=168,
        causal_max_hops=3
    )
)

# Add memories (all dimensions indexed automatically)
memory.add("The project deadline was extended to March")
memory.add("Budget was cut due to Q4 results")
memory.add("Team morale dropped after the layoffs")

# Search with automatic intent detection
results = memory.search("Why is team morale low?")
# Intent: CAUSAL → prioritizes causal graph traversal
# Returns: layoffs → morale, with causal chain

results = memory.search("What happened last month?")
# Intent: TEMPORAL → prioritizes time-based retrieval

results = memory.search("What do we know about the project?")
# Intent: ENTITY → prioritizes entity graph

# Explicit causal queries
chain = memory.get_causes("memory_id_123")
effects = memory.get_effects("memory_id_456")

# Entity queries
memories = memory.get_entity_memories("Project Alpha")

# Clean shutdown
memory.close()
```

---

## Target Users

### Primary: AI Application Developers

Developers building:
- Personal AI assistants with long-term memory
- Enterprise chatbots that remember context
- AI agents requiring persistent state
- Research tools with knowledge accumulation

**Their current pain:**
- Deploying and managing multiple databases
- Writing custom fusion logic
- Debugging across distributed systems
- Explaining to ops teams why they need 4 databases

**MnemeFusion value:**
- One file, one dependency
- Works offline, works embedded
- No infrastructure to manage
- Deterministic, debuggable

### Secondary: Edge AI / On-Device

Applications requiring:
- Offline-capable AI memory
- Privacy-preserving local storage
- Low-latency retrieval
- Minimal resource footprint

---

## Scaling Model

MnemeFusion is designed for **per-user embedded deployment**, not centralized multi-tenant.

### How It Scales

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│              (Developer's responsibility)                    │
│                                                              │
│   Load Balancer → Server Fleet → Object Storage             │
│                                                              │
│   User routing, sharding, replication, backup               │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Each user gets their own
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MnemeFusion Engine                        │
│                  (Our responsibility)                        │
│                                                              │
│   user_123.mfdb    user_456.mfdb    user_789.mfdb          │
│   (10MB-1GB)       (10MB-1GB)       (10MB-1GB)             │
│                                                              │
│   Fast, correct, reliable single-user operations            │
└─────────────────────────────────────────────────────────────┘
```

### Same Pattern As

| Product | Per-User Model | Scale Achieved |
|---------|---------------|----------------|
| SQLite | One DB per user/device | Billions of deployments |
| LevelDB | One store per instance | Used by Chrome, Bitcoin |
| Realm | One DB per mobile app | Millions of apps |

**We don't solve distributed systems. We solve single-user memory so well that horizontal scaling becomes trivial.**

---

## Competitive Position

### Current Landscape

| Product | Type | Dimensions | Deployment |
|---------|------|------------|------------|
| Qdrant | Vector DB | Semantic only | Server/Cloud |
| Pinecone | Vector DB | Semantic only | Cloud only |
| Neo4j | Graph DB | Entity/Causal | Server |
| SQLite | Relational | Temporal (manual) | Embedded |
| Weaviate | Vector DB | Semantic + refs | Server |

**Gap: No embedded engine with native 4D indexing.**

### MnemeFusion Position

```
                    Embedded ◄─────────────────► Server/Cloud
                        │                              │
                        │                              │
    Single Dimension    │   SQLite                     │   Qdrant
                        │   LevelDB                    │   Pinecone
                        │   redb                       │   Neo4j
                        │                              │
                        │                              │
                        │                              │
    Multi-Dimension     │   ┌─────────────┐            │   Weaviate
                        │   │ MnemeFusion │            │   (partial)
                        │   │   (GAP)     │            │
                        │   └─────────────┘            │
                        │                              │
```

**MnemeFusion owns the embedded + multi-dimensional quadrant.**

---

## Development Phases

### Phase 1: Core Engine (Months 1-4)

**Goal:** Working engine with all four dimensions, Python bindings, basic query planner.

| Month | Focus | Deliverable |
|-------|-------|-------------|
| 1 | Foundation | Rust project structure, storage layer (redb), basic file format |
| 2 | Indexes | HNSW integration (usearch), temporal B-tree, payload storage |
| 3 | Graphs | Causal graph, entity graph, persistence to storage layer |
| 4 | Query Planner | Intent classification, fusion algorithm, Python bindings (PyO3) |

**Exit Criteria:**
- `pip install mnemefusion` works
- All four dimensions functional
- Basic benchmarks passing
- Documentation for core API

### Phase 2: Production Hardening (Months 5-7)

**Goal:** Production-ready reliability, performance optimization, comprehensive testing.

| Month | Focus | Deliverable |
|-------|-------|-------------|
| 5 | Reliability | ACID guarantees, crash recovery, corruption detection |
| 6 | Performance | Query optimization, caching, batch operations |
| 7 | Polish | Comprehensive tests, documentation, examples, benchmarks |

**Exit Criteria:**
- Passes reliability test suite
- Performance benchmarks published
- API stable (1.0 candidate)
- Example applications

### Phase 3: Ecosystem (Months 8+)

**Goal:** Adoption, community, enterprise features.

- Open source release
- Community building
- Enterprise features (encryption, audit logging)
- Additional language bindings (Node.js, Go)
- Potential: managed cloud offering

---

## Success Metrics

### Phase 1 Success

| Metric | Target |
|--------|--------|
| Core functionality | All 4 dimensions working |
| Python bindings | Installable via pip |
| Basic performance | < 10ms search latency for 100K memories |
| Documentation | API reference complete |

### Phase 2 Success

| Metric | Target |
|--------|--------|
| Reliability | Zero data loss in chaos testing |
| Performance | < 5ms p99 search latency |
| Test coverage | > 80% |
| Benchmarks | Published comparison vs alternatives |

### Long-term Success

| Metric | Target |
|--------|--------|
| GitHub stars | 1K+ (traction signal) |
| PyPI downloads | 10K+ monthly |
| Production users | 5+ confirmed |
| Contributors | External contributions accepted |

---

## Risk Analysis

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Rust learning curve | Medium | Medium | Leverage existing libraries, don't over-engineer |
| Library limitations | Low | High | Evaluate alternatives early, design for swappability |
| Performance gaps | Medium | Medium | Profile continuously, optimize hot paths |
| PyO3 complexity | Low | Medium | Start with simple bindings, iterate |

### Market Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Competitors ship first | Medium | High | Focus on differentiation (4D native), not features |
| AI memory becomes commoditized | Medium | Medium | Deep integration, developer experience |
| Adoption slower than expected | Medium | Medium | Build for own use case first, dogfood |

### Resource Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scope creep | High | Medium | Strict phase gates, MVP discipline |
| Time constraints | Medium | Medium | Realistic timeline, cut scope not quality |
| Burnout | Medium | High | Sustainable pace, clear milestones |

---

## Open Questions

### Resolved

| Question | Decision |
|----------|----------|
| Build vs buy algorithms | Compose existing Rust libraries |
| Scaling model | Per-user embedded, not distributed |
| Primary language | Rust core, Python bindings |

### To Resolve in Phase 1

| Question | Options | Decision Point |
|----------|---------|----------------|
| Vector library | usearch vs hora vs custom | Month 1 evaluation |
| Graph persistence | petgraph + redb vs cozo | Month 3 evaluation |
| File format | Custom vs leverage redb | Month 1 design |
| Embedding handling | Built-in vs external | Month 4 API design |

---

## Resource Requirements

### Skills Needed

| Skill | Required Level | Current |
|-------|---------------|---------|
| Rust programming | Intermediate | Learning |
| Database internals | Conceptual | Sufficient |
| Python packaging | Basic | Sufficient |
| Algorithm implementation | Conceptual | Sufficient (using libraries) |

### Time Investment

| Phase | Duration | Hours/Week | Total Hours |
|-------|----------|------------|-------------|
| Phase 1 | 4 months | 15-20 | 260-350 |
| Phase 2 | 3 months | 15-20 | 195-260 |

### Dependencies

- Rust toolchain
- Python 3.11+
- Development machine with sufficient RAM for testing

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-13 | Separate product from Mneme | Different scope, timeline, technical requirements |
| 2025-01-13 | Compose Rust libraries | Faster to market, proven algorithms |
| 2025-01-13 | Per-user scaling model | Matches SQLite pattern, simpler architecture |
| 2025-01-13 | Python-first bindings | Largest AI/ML ecosystem |

---

## References

- State of Applied AI 2025 Report (validates RAG + memory architecture needs)
- usearch documentation
- redb documentation  
- petgraph documentation
- PyO3 user guide
- SQLite architecture documentation (inspiration)

---

## Notes

MnemeFusion originated from the observation that while building Mneme (Python library with external backends), the ideal end state would be a unified engine that doesn't require users to deploy multiple databases. Rather than pivot Mneme, MnemeFusion is a separate, more ambitious project that could eventually provide a backend for Mneme or serve as a standalone product.

The key insight: developers don't want to manage Qdrant + Neo4j + SQLite. They want `memory = Memory("./brain.mfdb")` and everything just works.
