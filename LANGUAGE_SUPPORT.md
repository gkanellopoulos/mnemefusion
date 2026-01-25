# Language Support in MnemeFusion

**Last Updated**: January 24, 2026

## Overview

MnemeFusion's **core functionality is language-agnostic** and works with any language using multilingual embedding models. However, two optional features currently support English only.

## Core Features: Language-Agnostic ✅

These features work with **any language**:

| Feature | Language Support | Implementation |
|---------|------------------|----------------|
| **Vector Search** | ✅ All languages | HNSW index works with any embedding vectors |
| **Temporal Indexing** | ✅ All languages | Timestamp-based, no text processing |
| **Causal Links** | ✅ All languages | Explicit relationship tracking via API |
| **Metadata Filtering** | ✅ All languages | Key-value based, UTF-8 strings |
| **Namespaces** | ✅ All languages | String-based namespace isolation |
| **Deduplication** | ✅ All languages | Vector similarity comparison |
| **Batch Operations** | ✅ All languages | Bulk insert/update operations |
| **ACID Transactions** | ✅ All languages | Database-level guarantees |

## Optional Features: English-Only ⚠️

These features are currently English-only but can be disabled:

### 1. Entity Extraction

**Status**: English-only
**Location**: `src/ingest/entity_extractor.rs`

**What it does**:
- Extracts entities from text content (e.g., "Alice", "Project Alpha")
- Builds entity graph linking entities to memories
- Enables entity-based queries

**Why English-only**:
- Uses English stop words list ("The", "A", "Monday", etc.)
- Relies on capitalization rules (works for English/European languages)
- Not suitable for Chinese, Japanese, Arabic, etc.

**Impact when disabled**:
- ✅ Semantic search still works (embeddings capture entity mentions)
- ⚠️ No dedicated entity graph
- ⚠️ Entity queries less efficient (but still functional via semantic search)

**Workaround**:
```rust
// Disable entity extraction for non-English content
let config = Config::new()
    .with_entity_extraction(false);

// Or use your own NER pipeline and add entities manually
```

### 2. Intent Classification

**Status**: English-only
**Location**: `src/query/intent.rs`

**What it does**:
- Classifies queries into intents (Temporal, Causal, Entity, Factual)
- Optimizes fusion weights based on intent
- Example: "Why did X happen?" → Causal intent → prioritize causal dimension

**Why English-only**:
- Uses English regex patterns ("why", "yesterday", "about", etc.)
- Month names, weekday names in English
- Causal keywords in English

**Impact for non-English queries**:
- ✅ Query still works (semantic search always active)
- ⚠️ Falls back to "Factual" intent (pure semantic search)
- ⚠️ Suboptimal fusion weights (doesn't prioritize correct dimension)
- ❌ Example: "为什么会议被取消？" (Chinese "Why was meeting cancelled?") won't be classified as Causal

**Workaround**:
```rust
// Intent classification is always active but fails gracefully
// Non-English queries default to Factual intent (semantic search)
// This is acceptable for most use cases
```

## Multilingual Usage Guide

### Step 1: Choose a Multilingual Embedding Model

Recommended models:

| Model | Languages | Dimensions | Provider |
|-------|-----------|------------|----------|
| paraphrase-multilingual-MiniLM-L12-v2 | 50+ | 384 | HuggingFace |
| multilingual-e5-base | 100+ | 768 | HuggingFace |
| multilingual-e5-large | 100+ | 1024 | HuggingFace |
| text-embedding-3-small | 100+ | 1536 | OpenAI API |
| text-embedding-3-large | 100+ | 3072 | OpenAI API |

### Step 2: Configure MnemeFusion

```rust
use mnemefusion_core::{MemoryEngine, Config};

let config = Config::new()
    .with_embedding_dim(768)  // Match your model
    .with_entity_extraction(false);  // Disable English-only feature

let engine = MemoryEngine::open("brain.mfdb", config)?;
```

### Step 3: Use Multilingual Embeddings

**Python Example** (Chinese):

```python
from sentence_transformers import SentenceTransformer
import mnemefusion

# Load multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Configure MnemeFusion
config = mnemefusion.Config()
config.entity_extraction_enabled = False
config.embedding_dim = 384

memory = mnemefusion.Memory("brain.mfdb", config)

# Add Chinese memory
text = "我今天学习了机器学习算法"
embedding = model.encode(text)
memory.add(text, embedding.tolist())

# Search in Chinese
query = "机器学习"
query_emb = model.encode(query)
results = memory.search(query_emb.tolist(), top_k=10)
# ✅ Works perfectly! Finds semantically similar memories
```

**Python Example** (Mixed Languages):

```python
# Add memories in different languages
memory.add("Alice completed the project", model.encode("...").tolist())
memory.add("Alice完成了项目", model.encode("...").tolist())
memory.add("Alice terminó el proyecto", model.encode("...").tolist())

# Search works across languages with multilingual embeddings
results = memory.search(model.encode("project completion").tolist(), top_k=10)
# ✅ Finds all three memories regardless of language
```

## What Still Works Without English

Even with entity extraction disabled and intent classification falling back to English:

### Fully Functional ✅

1. **Semantic Search** - Core value proposition, works perfectly
2. **Temporal Queries** - Time-based filtering and recency search
3. **Metadata Filtering** - Filter by custom metadata fields
4. **Namespaces** - Isolate memories by namespace
5. **Deduplication** - Detect duplicate content via similarity
6. **Batch Operations** - Efficient bulk operations
7. **Causal Relationships** - Track via explicit API calls

### Degraded Performance ⚠️

1. **Entity Queries** - No dedicated entity graph, but semantic search compensates
2. **Intent-Optimized Fusion** - Falls back to semantic-heavy weights (still works)

### Not Available ❌

1. **Automatic Entity Extraction** - Must use your own NER pipeline
2. **Multi-language Intent Classification** - Always uses Factual intent

## Future Roadmap

### Phase 4: Multilingual Support (Future)

**Planned improvements**:

1. **Pluggable Entity Extractors**
   - Trait-based design already in place
   - Add language-specific implementations
   - Support spaCy multilingual NER
   - Support custom user-provided extractors

2. **Pluggable Intent Classifiers**
   - Language detection
   - Language-specific pattern libraries
   - ML-based intent classification (language-agnostic)

3. **Language Configuration**
   ```rust
   Config::new()
       .with_language("zh-CN")  // Chinese
       .with_entity_extractor(ChineseNER::new())
       .with_intent_classifier(MultilingualIntentClassifier::new())
   ```

### Contributing

We welcome contributions for multilingual support! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**High-priority contributions**:
- Language-specific entity extractors (spaCy integration)
- ML-based intent classification (language-agnostic)
- Multilingual test cases
- Documentation translations

## FAQ

### Q: Can I use MnemeFusion with Chinese/Arabic/Japanese?

**A: Yes!** The core vector search works perfectly with multilingual embeddings. Just disable entity extraction:

```rust
Config::new().with_entity_extraction(false)
```

### Q: Will my queries work in non-English languages?

**A: Yes, with caveats:**
- ✅ Semantic search works perfectly
- ⚠️ Intent classification defaults to Factual (semantic-heavy weights)
- ⚠️ Entity-based queries won't use entity graph (but semantic search compensates)

### Q: What embedding models support multiple languages?

**A:** Most modern models support 50-100+ languages:
- HuggingFace: `paraphrase-multilingual-*`, `multilingual-e5-*`
- OpenAI: `text-embedding-3-small`, `text-embedding-3-large`
- Cohere: `embed-multilingual-*`

### Q: Can I use my own NER pipeline?

**A: Yes!** Extract entities yourself and add them via the API:

```rust
// Your custom NER pipeline
let entities = my_ner_pipeline.extract(content);

// Add memory without automatic extraction
let id = engine.add(content, embedding, metadata, timestamp, None, None)?;

// Manually add entities
for entity_name in entities {
    engine.add_entity_mention(&id, &entity_name)?;
}
```

### Q: When will multilingual entity/intent support be added?

**A:** Tentatively Phase 4 (post-1.0 release). The current design uses traits to make this pluggable, so it can be added without breaking changes.

### Q: Does this affect performance?

**A:** No. Disabling entity extraction actually improves performance slightly:
- Removes ~0.3ms from add operations (entity extraction overhead)
- No impact on search performance

---

**For questions or contributions, see**:
- [GitHub Issues](https://github.com/gkanellopoulos/mnemefusion/issues)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Documentation](https://docs.rs/mnemefusion-core)
