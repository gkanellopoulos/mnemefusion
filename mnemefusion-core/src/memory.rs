//! Memory engine - main API entry point
//!
//! The MemoryEngine struct provides the primary interface for interacting
//! with a MnemeFusion database.

use crate::{
    config::Config,
    error::{Error, Result},
    graph::{CausalTraversalResult, GraphManager},
    index::{BM25Config, BM25Index, TemporalIndex, VectorIndex, VectorIndexConfig},
    ingest::IngestionPipeline,
    query::{profile_search::fact_embedding_key, FusedResult, IntentClassification, QueryPlanner},
    storage::StorageEngine,
    types::{
        AddResult, BatchResult, Entity, EntityProfile, Memory, MemoryId, MemoryInput,
        MetadataFilter, Source, Timestamp, UpsertResult,
    },
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

#[cfg(feature = "slm")]
use crate::ingest::SlmMetadataExtractor;
#[cfg(feature = "slm")]
use std::sync::Mutex;

#[cfg(feature = "entity-extraction")]
use crate::extraction::{LlmEntityExtractor, ModelTier};
#[cfg(all(feature = "entity-extraction", not(feature = "slm")))]
use std::sync::Mutex;

/// Callback type for computing embeddings at ingestion time.
///
/// Provided by the caller (e.g., Python's `SentenceTransformer.encode()`).
/// Called for each fact text during ingestion to compute fact embeddings.
pub type EmbeddingFn = Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>;

/// Build contextual text for embedding by prepending speaker metadata.
///
/// When computing embeddings, prepending the speaker name helps the embedding model
/// distinguish between the same statement made by different people. The original
/// content is stored as-is; only the embedding carries the speaker context.
///
/// Returns the original content unchanged if no "speaker" key exists in metadata.
pub fn contextualize_for_embedding(content: &str, metadata: &HashMap<String, String>) -> String {
    if let Some(speaker) = metadata.get("speaker") {
        if !speaker.is_empty() {
            return format!("{}: {}", speaker, content);
        }
    }
    content.to_string()
}

/// Main memory engine interface
///
/// This is the primary entry point for all MnemeFusion operations.
/// It coordinates storage, indexing, and retrieval across all dimensions.
pub struct MemoryEngine {
    storage: Arc<StorageEngine>,
    vector_index: Arc<RwLock<VectorIndex>>,
    bm25_index: Arc<BM25Index>,
    temporal_index: Arc<TemporalIndex>,
    graph_manager: Arc<RwLock<GraphManager>>,
    pipeline: IngestionPipeline,
    query_planner: QueryPlanner,
    config: Config,
    /// Auto-embedding engine. Set via `Config::embedding_model` or `with_embedding_engine()`.
    #[cfg(feature = "embedding-onnx")]
    embedding_engine: Option<std::sync::Arc<crate::embedding::EmbeddingEngine>>,
    /// Default namespace applied to all add/query calls when `namespace` arg is None.
    /// Set via `with_user(user_name)`.
    default_namespace: Option<String>,
}

impl MemoryEngine {
    /// Open or create a memory database
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .mfdb file
    /// * `config` - Configuration options
    ///
    /// # Returns
    ///
    /// A new MemoryEngine instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The database file cannot be created or opened
    /// - The file format is invalid
    /// - The configuration is invalid
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    ///
    /// let engine = MemoryEngine::open("./brain.mfdb", Config::default()).unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P, config: Config) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Open storage
        let storage = Arc::new(StorageEngine::open(path)?);

        // Create vector index configuration from main config
        let vector_config = VectorIndexConfig {
            dimension: config.embedding_dim,
            connectivity: config.hnsw_m,
            expansion_add: config.hnsw_ef_construction,
            expansion_search: config.hnsw_ef_search,
        };

        // Create and load vector index
        let mut vector_index = VectorIndex::new(vector_config, Arc::clone(&storage))?;
        vector_index.load()?;
        let vector_index = Arc::new(RwLock::new(vector_index));

        // Create and load BM25 index
        let bm25_config = BM25Config::default();
        let bm25_index = Arc::new(BM25Index::new(Arc::clone(&storage), bm25_config));
        bm25_index.load()?;

        // Create temporal index
        let temporal_index = Arc::new(TemporalIndex::new(Arc::clone(&storage)));

        // Create and load graph manager
        let mut graph_manager = GraphManager::new();
        crate::graph::persist::load_graph(&mut graph_manager, &storage)?;

        // One-time migration: repair Entity→Entity edges from profile facts
        // (older DBs stored relationship facts but lost graph edges on save)
        crate::graph::persist::repair_relationship_edges(&mut graph_manager, &storage)?;

        let graph_manager = Arc::new(RwLock::new(graph_manager));

        // Create ingestion pipeline
        let mut pipeline = IngestionPipeline::new(
            Arc::clone(&storage),
            Arc::clone(&vector_index),
            Arc::clone(&bm25_index),
            Arc::clone(&temporal_index),
            Arc::clone(&graph_manager),
            config.entity_extraction_enabled,
        );

        // Attach SLM metadata extractor if enabled
        #[cfg(feature = "slm")]
        if config.slm_metadata_extraction_enabled {
            if let Some(ref slm_config) = config.slm_config {
                tracing::info!("Initializing SLM metadata extractor for ingestion...");
                match SlmMetadataExtractor::new(slm_config.clone()) {
                    Ok(extractor) => {
                        pipeline = pipeline.with_slm_extractor(Arc::new(Mutex::new(extractor)));
                        tracing::info!("SLM metadata extractor attached to pipeline");
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to initialize SLM metadata extractor, using pattern-based extraction: {}",
                            e
                        );
                    }
                }
            } else {
                tracing::debug!("SLM metadata extraction enabled but no slm_config provided");
            }
        }

        // Wire extraction_passes from config to pipeline
        #[cfg(feature = "entity-extraction")]
        if config.extraction_passes > 1 {
            pipeline = pipeline.with_extraction_passes(config.extraction_passes);
        }

        // Wire profile entity type filter from config to pipeline
        pipeline.set_profile_entity_types(config.profile_entity_types.clone());

        // Wire async extraction threshold from config to pipeline
        if config.async_extraction_threshold > 0 {
            pipeline.set_async_extraction_threshold(config.async_extraction_threshold);
        }

        // Create query planner
        let query_planner = QueryPlanner::new(
            Arc::clone(&storage),
            Arc::clone(&vector_index),
            Arc::clone(&bm25_index),
            Arc::clone(&temporal_index),
            Arc::clone(&graph_manager),
            config.fusion_semantic_threshold,
            config.semantic_prefilter_threshold,
            config.fusion_strategy,
            config.rrf_k,
            #[cfg(feature = "slm")]
            config.slm_config.clone(),
            config.slm_query_classification_enabled,
            config.adaptive_k_threshold,
        )?;

        // Initialize embedding engine if configured
        #[cfg(feature = "embedding-onnx")]
        let embedding_engine = if let Some(ref model_path) = config.embedding_model {
            tracing::info!("Initializing embedding engine from '{}'...", model_path);
            match crate::embedding::EmbeddingEngine::from_path(model_path) {
                Ok(engine) => {
                    tracing::info!("Embedding engine ready (dim={})", engine.dim);
                    Some(std::sync::Arc::new(engine))
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load embedding model from '{}': {}. \
                         Embeddings must be supplied explicitly.",
                        model_path,
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            storage,
            vector_index,
            bm25_index,
            temporal_index,
            graph_manager,
            pipeline,
            query_planner,
            config,
            #[cfg(feature = "embedding-onnx")]
            embedding_engine,
            default_namespace: None,
        })
    }

    /// Enable native LLM entity extraction with the specified model tier
    ///
    /// This enables automatic entity and fact extraction during memory ingestion
    /// using a locally-running LLM via llama.cpp. Extraction results are stored
    /// in entity profiles for fast retrieval.
    ///
    /// # Arguments
    ///
    /// * `tier` - Model tier to use (Balanced = 4B, Quality = 7B)
    ///
    /// # Returns
    ///
    /// Self for method chaining
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config, ModelTier};
    ///
    /// let engine = MemoryEngine::open("./brain.mfdb", Config::default())?
    ///     .with_llm_entity_extraction(ModelTier::Balanced)?;
    /// # Ok::<(), mnemefusion_core::Error>(())
    /// ```
    #[cfg(feature = "entity-extraction")]
    pub fn with_llm_entity_extraction(mut self, tier: ModelTier) -> Result<Self> {
        tracing::info!("Initializing LLM entity extractor ({:?})...", tier);

        let extractor = LlmEntityExtractor::load(tier)?;
        self.pipeline = self
            .pipeline
            .with_llm_extractor(Arc::new(Mutex::new(extractor)));

        tracing::info!("LLM entity extractor attached to pipeline");
        Ok(self)
    }

    /// Enable native LLM entity extraction with a custom model path
    ///
    /// This enables automatic entity and fact extraction using a model
    /// at the specified path.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the GGUF model file
    /// * `tier` - Model tier (affects generation parameters)
    ///
    /// # Returns
    ///
    /// Self for method chaining
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    #[cfg(feature = "entity-extraction")]
    pub fn with_llm_entity_extraction_from_path(
        mut self,
        model_path: impl Into<std::path::PathBuf>,
        tier: ModelTier,
    ) -> Result<Self> {
        tracing::info!("Initializing LLM entity extractor from custom path...");

        let extractor = LlmEntityExtractor::load_from_path(model_path, tier)?;
        self.pipeline = self
            .pipeline
            .with_llm_extractor(Arc::new(Mutex::new(extractor)));

        tracing::info!("LLM entity extractor attached to pipeline");
        Ok(self)
    }

    /// Set the number of LLM extraction passes per document.
    ///
    /// This must be called after `with_llm_entity_extraction*()` to take effect.
    /// Multiple passes capture different facts, producing richer profiles.
    #[cfg(feature = "entity-extraction")]
    pub fn set_extraction_passes(&mut self, passes: usize) {
        self.pipeline.set_extraction_passes(passes);
    }

    /// Process all deferred LLM extractions queued by `add()` in async mode.
    ///
    /// When `async_extraction_threshold > 0` (set via config or
    /// `with_async_extraction_threshold()`), `add()` stores large memories
    /// immediately and defers LLM extraction here. Call this periodically
    /// (e.g., every N messages, or before querying) to build entity profiles.
    ///
    /// Returns the number of memories whose extraction was processed.
    /// Safe to call when the queue is empty (returns `Ok(0)`).
    pub fn flush_extraction_queue(&self) -> Result<usize> {
        self.pipeline.flush_extraction_queue()
    }

    /// Returns the number of memories with deferred LLM extractions pending.
    ///
    /// Non-zero only when `async_extraction_threshold > 0` and large `add()` calls
    /// have been made since the last `flush_extraction_queue()`.
    pub fn pending_extraction_count(&self) -> usize {
        self.pipeline.pending_extraction_count()
    }

    /// Set a default namespace (user identity) for all add/query operations.
    ///
    /// When set, any call to `add()` or `query()` that does not supply an explicit
    /// `namespace` argument will use this value automatically. Equivalent to always
    /// passing `namespace = Some(user)` — enables "Memory is per-user" semantics
    /// without changing every call site.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// let engine = MemoryEngine::open("./brain.mfdb", Config::default()).unwrap()
    ///     .with_user("alice");
    /// // All subsequent add/query calls default to namespace="alice"
    /// ```
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.default_namespace = Some(user.into());
        self
    }

    /// Attach an embedding engine for automatic text vectorization.
    ///
    /// After this call, `add()` and `query()` can be called without supplying
    /// explicit embedding vectors.
    ///
    /// Requires the `embedding-onnx` feature at compile time.
    #[cfg(feature = "embedding-onnx")]
    pub fn with_embedding_engine(
        mut self,
        engine: crate::embedding::EmbeddingEngine,
    ) -> Self {
        self.embedding_engine = Some(std::sync::Arc::new(engine));
        self
    }

    /// Auto-compute an embedding using the configured engine.
    ///
    /// Returns `Err(Error::NoEmbeddingEngine)` if no engine is configured.
    #[cfg(feature = "embedding-onnx")]
    fn auto_embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embedding_engine
            .as_ref()
            .ok_or(Error::NoEmbeddingEngine)?
            .embed(text)
    }

    /// Auto-compute an embedding (always errors when feature is disabled).
    #[cfg(not(feature = "embedding-onnx"))]
    fn auto_embed(&self, _text: &str) -> Result<Vec<f32>> {
        Err(Error::NoEmbeddingEngine)
    }

    /// Set the embedding function for computing fact embeddings at ingestion time.
    ///
    /// When set, the pipeline will compute and store embeddings for each extracted
    /// entity fact during ingestion. These embeddings enable semantic matching in
    /// ProfileSearch (cosine similarity vs word-overlap).
    ///
    /// The function should return an embedding vector for the given text input.
    /// Typically this wraps the same embedding model used for memory embeddings
    /// (e.g., `SentenceTransformer.encode()`).
    ///
    /// # Arguments
    ///
    /// * `f` - Embedding function: `Fn(&str) -> Vec<f32>`
    pub fn set_embedding_fn(&mut self, f: EmbeddingFn) {
        self.pipeline.set_embedding_fn(f);
    }

    /// Precompute missing fact embeddings for all entity profiles.
    ///
    /// Iterates all stored profiles, checks each fact for a stored embedding,
    /// and computes + stores any missing ones using the registered EmbeddingFn.
    /// This is a one-time backfill operation — "pay the cost once."
    ///
    /// Returns the number of fact embeddings computed.
    pub fn precompute_fact_embeddings(&self) -> Result<usize> {
        let embed_fn = self
            .pipeline
            .embedding_fn()
            .ok_or_else(|| {
                Error::Configuration(
                    "No embedding function set. Call set_embedding_fn() first.".into(),
                )
            })?;

        let profiles = self.storage.list_entity_profiles()?;
        let mut computed = 0;

        for profile in &profiles {
            for (fact_type, facts) in &profile.facts {
                for fact in facts {
                    let key = fact_embedding_key(&profile.name, fact_type, &fact.value);
                    if self.storage.get_fact_embedding(&key)?.is_none() {
                        let fact_text =
                            format!("{} {}", fact_type.replace('_', " "), fact.value);
                        let embedding = embed_fn(&fact_text);
                        self.storage.store_fact_embedding(&key, &embedding)?;
                        computed += 1;
                    }
                }
            }
        }

        Ok(computed)
    }

    /// Run entity extraction on text without adding to the database.
    ///
    /// Useful for testing extraction quality or comparing model outputs.
    /// Requires `with_llm_entity_extraction*()` to have been called first.
    ///
    /// # Arguments
    /// * `content` - The text to extract entities from
    /// * `speaker` - Optional speaker name for first-person attribution
    ///
    /// # Returns
    /// The extraction result with entities, facts, records, and relationships.
    #[cfg(feature = "entity-extraction")]
    pub fn extract_text(
        &self,
        content: &str,
        speaker: Option<&str>,
    ) -> Result<crate::extraction::ExtractionResult> {
        self.pipeline.extract_text(content, speaker)
    }

    /// Apply an externally-produced extraction result to a memory's entity profiles.
    ///
    /// This enables API-based extraction backends (e.g., NScale cloud inference)
    /// to inject entity profiles without requiring a local LLM. The extraction
    /// result must match the same JSON schema as the local Qwen3 extractor.
    ///
    /// # Arguments
    /// * `memory_id` - The memory ID to associate the extraction with
    /// * `extraction` - The extraction result from an external source
    #[cfg(feature = "entity-extraction")]
    pub fn apply_extraction(
        &self,
        memory_id: &MemoryId,
        extraction: &crate::extraction::ExtractionResult,
    ) -> Result<()> {
        // Update entity profiles from facts
        self.pipeline
            .update_entity_profiles_from_llm(memory_id, extraction)?;

        // Store entity-to-entity relationships
        if !extraction.relationships.is_empty() {
            self.pipeline
                .store_relationships(memory_id, &extraction.relationships)?;
        }

        // Annotate parent memory with typed record metadata (record_type, event_date)
        // Note: We do NOT create child memories — they flood the vector index and
        // cause recall collapse (-14.9 pts in S30 testing). Instead, typed decomposition
        // is stored as metadata on the parent for type-aware retrieval balancing.
        if !extraction.records.is_empty() {
            self.pipeline
                .annotate_parent_with_types(memory_id, &extraction.records);
        }

        Ok(())
    }

    /// Generate summaries for all entity profiles.
    ///
    /// For each profile with facts, generates a dense summary paragraph that
    /// condenses the profile's facts into one text block. When present, query()
    /// injects summaries as single context items instead of N individual facts,
    /// addressing RANK failures where evidence is present but buried.
    ///
    /// Returns the number of profiles summarized.
    pub fn summarize_profiles(&self) -> Result<usize> {
        let profiles = self.storage.list_entity_profiles()?;
        let mut summarized = 0;
        for mut profile in profiles {
            if profile.generate_summary().is_some() {
                self.storage.store_entity_profile(&profile)?;
                summarized += 1;
            }
        }
        Ok(summarized)
    }

    /// Consolidate entity profiles by removing noise and deduplicating facts.
    ///
    /// Performs the following cleanup operations:
    /// 1. Remove null-indicator values ("none", "N/A", etc.)
    /// 2. Remove overly verbose values (>100 chars)
    /// 3. Semantic dedup within same fact_type using embedding similarity (threshold: 0.85)
    ///    — keeps fact with higher confidence, or first encountered on tie
    /// 4. Delete garbage entity profiles (non-person entities with ≤2 facts)
    ///
    /// Returns (facts_removed, profiles_deleted).
    pub fn consolidate_profiles(&self) -> Result<(usize, usize)> {
        use crate::query::profile_search::{cosine_similarity, resolve_entity_alias};

        let embed_fn = self.pipeline.embedding_fn();

        let mut total_facts_removed = 0usize;
        let mut profiles_deleted = 0usize;

        // Phase 0: Merge alias profiles into their canonical forms.
        // E.g., "mel" → "melanie", "mell" → "melanie" (via fuzzy matching).
        {
            let mut all_names = self.storage.list_entity_profile_names()?;
            // Sort by length (shortest first) so short aliases resolve to longer canonicals
            all_names.sort_by_key(|n| n.len());

            let mut merged_away: std::collections::HashSet<String> = std::collections::HashSet::new();

            for i in 0..all_names.len() {
                let short_name = &all_names[i];
                if merged_away.contains(short_name) {
                    continue;
                }
                if let Some(canonical) = resolve_entity_alias(short_name, &all_names) {
                    if merged_away.contains(&canonical) {
                        continue;
                    }
                    // Load both profiles
                    let short_profile = match self.storage.get_entity_profile(short_name)? {
                        Some(p) => p,
                        None => continue,
                    };
                    let mut canon_profile = match self.storage.get_entity_profile(&canonical)? {
                        Some(p) => p,
                        None => continue,
                    };

                    // Move all facts from short → canonical (add_fact handles dedup)
                    for (_fact_type, facts) in &short_profile.facts {
                        for fact in facts {
                            canon_profile.add_fact(fact.clone());
                        }
                    }

                    // Move all source_memories
                    for mem_id in &short_profile.source_memories {
                        canon_profile.add_source_memory(mem_id.clone());
                    }

                    // Save canonical, delete alias
                    self.storage.store_entity_profile(&canon_profile)?;
                    self.storage.delete_entity_profile(short_name)?;
                    merged_away.insert(short_name.clone());
                    profiles_deleted += 1;

                    tracing::info!(
                        "Merged alias profile '{}' into canonical '{}'",
                        short_name,
                        canonical,
                    );
                }
            }
        }

        let profiles = self.storage.list_entity_profiles()?;

        const NULL_INDICATORS: &[&str] = &[
            "none", "n/a", "na", "not specified", "not mentioned",
            "unknown", "unspecified", "not provided", "no information",
        ];

        for mut profile in profiles {
            let mut facts_removed_in_profile = 0usize;

            // Phase 1 & 2: Remove null and long values
            for (_fact_type, facts) in profile.facts.iter_mut() {
                let before = facts.len();
                facts.retain(|f| {
                    let trimmed = f.value.trim();
                    let lower = trimmed.to_lowercase();
                    // Keep if NOT a null indicator AND NOT too long
                    !NULL_INDICATORS.contains(&lower.as_str()) && trimmed.len() <= 100
                });
                facts_removed_in_profile += before - facts.len();
            }

            // Phase 3: Semantic dedup within same fact_type (requires embedding fn)
            if let Some(ref embed_fn) = embed_fn {
                for (fact_type, facts) in profile.facts.iter_mut() {
                    if facts.len() <= 1 {
                        continue;
                    }

                    // Sort by confidence descending (highest confidence kept first)
                    facts.sort_by(|a, b| {
                        b.confidence
                            .partial_cmp(&a.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    // Collect embeddings for all facts
                    let embeddings: Vec<Vec<f32>> = facts
                        .iter()
                        .map(|f| {
                            // Try stored embedding first, compute on the fly if missing
                            let key = fact_embedding_key(&profile.name, fact_type, &f.value);
                            self.storage
                                .get_fact_embedding(&key)
                                .ok()
                                .flatten()
                                .unwrap_or_else(|| {
                                    let text = format!("{} {}", fact_type.replace('_', " "), f.value);
                                    embed_fn(&text)
                                })
                        })
                        .collect();

                    // Greedy dedup: keep first (highest confidence), skip near-duplicates
                    let mut keep_indices: Vec<usize> = Vec::new();
                    for i in 0..facts.len() {
                        let mut is_dup = false;
                        for &kept_idx in &keep_indices {
                            let sim = cosine_similarity(&embeddings[i], &embeddings[kept_idx]);
                            if sim > 0.85 {
                                is_dup = true;
                                break;
                            }
                        }
                        if !is_dup {
                            keep_indices.push(i);
                        }
                    }

                    let before = facts.len();
                    let kept_facts: Vec<_> = keep_indices
                        .into_iter()
                        .map(|i| facts[i].clone())
                        .collect();
                    *facts = kept_facts;
                    facts_removed_in_profile += before - facts.len();
                }
            }

            // Remove empty fact type entries
            profile.facts.retain(|_, v| !v.is_empty());

            total_facts_removed += facts_removed_in_profile;

            // Phase 4: Delete garbage profiles (non-person with ≤2 facts)
            let total_facts = profile.total_facts();
            if profile.entity_type != "person" && total_facts <= 2 {
                self.storage.delete_entity_profile(&profile.name)?;
                profiles_deleted += 1;
                continue;
            }

            // Save updated profile if any facts were removed
            if facts_removed_in_profile > 0 {
                self.storage.store_entity_profile(&profile)?;
            }
        }

        Ok((total_facts_removed, profiles_deleted))
    }

    /// Repair entity profiles by re-processing llm_extraction metadata stored in memories.
    ///
    /// This is a recovery function for databases where entity profiles are missing or
    /// incomplete due to extraction failures, consolidation over-pruning, or ingestion bugs.
    ///
    /// For every memory in the DB:
    /// 1. Parse the `llm_extraction` JSON from metadata (if present)
    /// 2. For each entity_fact: create/update the entity profile with the fact
    ///    and add the memory as a source_memory
    /// 3. For the `speaker` metadata field: ensure the speaker entity's profile
    ///    includes this memory as a source_memory (handles first-person statements
    ///    where the speaker name isn't in the content text)
    ///
    /// Respects the pipeline's `profile_entity_types` filter and type allowlist.
    /// Skips entities whose names appear to be pronouns or generic placeholders.
    ///
    /// Returns (profiles_created, source_memories_added).
    pub fn repair_profiles_from_metadata(&self) -> Result<(usize, usize)> {
        use crate::types::{EntityFact, EntityId};
        use crate::query::profile_search::resolve_entity_alias;

        let junk_names: &[&str] = &[
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
            "they", "them", "his", "her", "their", "him", "this", "that",
            "unknown", "unspecified", "someone", "somebody", "anyone",
        ];

        let allowed_types: &[&str] = &["person", "organization", "location"];

        let mut profiles_created = 0usize;
        let mut source_memories_added = 0usize;

        let all_ids = self.storage.list_memory_ids()?;
        let total = all_ids.len();
        tracing::info!("repair_profiles_from_metadata: scanning {} memories", total);

        for (idx, mem_id) in all_ids.iter().enumerate() {
            if idx % 500 == 0 {
                tracing::info!("  {}/{}", idx, total);
            }

            let memory = match self.storage.get_memory(mem_id)? {
                Some(m) => m,
                None => continue,
            };

            // Load fresh known_names once per memory (profiles may have been added)
            let known_names = self.storage.list_entity_profile_names()?;

            // ── Step A: re-process llm_extraction entity_facts ──────────────────
            if let Some(json_str) = memory.metadata.get("llm_extraction") {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                    // Build entity_type lookup from "entities" array
                    let mut entity_types: HashMap<String, String> = HashMap::new();
                    if let Some(ents) = v["entities"].as_array() {
                        for e in ents {
                            if let (Some(name), Some(etype)) =
                                (e["name"].as_str(), e["type"].as_str())
                            {
                                entity_types
                                    .insert(name.to_lowercase(), etype.to_lowercase());
                            }
                        }
                    }

                    if let Some(facts_arr) = v["entity_facts"].as_array() {
                        for fact_val in facts_arr {
                            let entity_raw = match fact_val["entity"].as_str() {
                                Some(e) => e,
                                None => continue,
                            };
                            let entity_lower = entity_raw.to_lowercase();

                            // Skip junk names and single-char names
                            if entity_lower.len() < 2
                                || junk_names.contains(&entity_lower.as_str())
                            {
                                continue;
                            }

                            let etype = entity_types
                                .get(&entity_lower)
                                .map(|s| s.as_str())
                                .unwrap_or("person");

                            // Only create profiles for allowed entity types
                            if !allowed_types.contains(&etype) {
                                continue;
                            }

                            // Canonicalize via alias resolution
                            let canonical = resolve_entity_alias(&entity_lower, &known_names)
                                .unwrap_or_else(|| entity_lower.clone());

                            let fact_type = fact_val["fact_type"]
                                .as_str()
                                .unwrap_or("unknown")
                                .to_string();
                            let value = fact_val["value"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            let confidence = fact_val["confidence"]
                                .as_f64()
                                .unwrap_or(0.8) as f32;

                            if value.is_empty() || value.len() > 100 {
                                continue;
                            }

                            let is_new =
                                self.storage.get_entity_profile(&canonical)?.is_none();

                            let mut profile = self
                                .storage
                                .get_entity_profile(&canonical)?
                                .unwrap_or_else(|| {
                                    EntityProfile::new(
                                        EntityId::new(),
                                        canonical.clone(),
                                        etype.to_string(),
                                    )
                                });

                            profile.add_fact(EntityFact {
                                fact_type,
                                value,
                                confidence,
                                source_memory: mem_id.clone(),
                                extracted_at: Timestamp::now(),
                            });
                            profile.add_source_memory(mem_id.clone());

                            self.storage.store_entity_profile(&profile)?;

                            if is_new {
                                profiles_created += 1;
                            }
                            source_memories_added += 1;
                        }
                    }
                }
            }

            // ── Step B: speaker → source_memory attribution ─────────────────────
            // When the speaker says "I joined a gym", the entity name ("Maria") isn't
            // in the content. If an entity profile exists for the speaker, add this
            // memory as a source_memory so query-time entity injection can find it.
            if let Some(speaker_raw) = memory.metadata.get("speaker") {
                let speaker_lower = speaker_raw.trim().to_lowercase();
                if speaker_lower.len() < 2 || junk_names.contains(&speaker_lower.as_str()) {
                    continue;
                }

                // Re-load known_names (may have been updated by Step A above)
                let known_names2 = self.storage.list_entity_profile_names()?;
                let canonical = resolve_entity_alias(&speaker_lower, &known_names2)
                    .unwrap_or_else(|| speaker_lower.clone());

                if let Ok(Some(mut profile)) = self.storage.get_entity_profile(&canonical) {
                    if !profile.source_memories.contains(mem_id) {
                        profile.add_source_memory(mem_id.clone());
                        self.storage.store_entity_profile(&profile)?;
                        source_memories_added += 1;
                    }
                }
            }
        }

        tracing::info!(
            "repair_profiles_from_metadata: created {} profiles, added {} source_memory links",
            profiles_created,
            source_memories_added
        );
        Ok((profiles_created, source_memories_added))
    }

    /// Add a new memory to the database
    ///
    /// This will automatically index the memory across all dimensions:
    /// - Semantic (vector similarity)
    /// - Temporal (time-based)
    /// - Entity (if auto-extraction enabled)
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to store
    /// * `embedding` - Vector embedding (must match configured dimension)
    /// * `metadata` - Optional key-value metadata
    /// * `timestamp` - Optional custom timestamp (defaults to now)
    /// * `source` - Optional provenance/source tracking information
    ///
    /// # Returns
    ///
    /// The ID of the created memory
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Embedding dimension doesn't match configuration
    /// - Storage operation fails
    /// - Source serialization fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # use mnemefusion_core::types::{Source, SourceType};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let embedding = vec![0.1; 384];
    ///
    /// // Add memory with source tracking
    /// let source = Source::new(SourceType::Conversation)
    ///     .with_id("conv_123")
    ///     .with_confidence(0.95);
    ///
    /// let id = engine.add(
    ///     "Meeting scheduled for next week".to_string(),
    ///     embedding,
    ///     None,
    ///     None,
    ///     Some(source),
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn add(
        &self,
        content: String,
        embedding: impl Into<Option<Vec<f32>>>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
        namespace: Option<&str>,
    ) -> Result<MemoryId> {
        // Resolve embedding: use provided value or auto-compute from content
        let embedding = match embedding.into() {
            Some(e) => e,
            None => self.auto_embed(&content)?,
        };

        // Validate embedding dimension
        if embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }

        // Apply default namespace if caller didn't supply one
        let effective_ns = namespace.or(self.default_namespace.as_deref());

        // Create memory
        let mut memory = if let Some(ts) = timestamp {
            let mut mem = Memory::new_with_timestamp(content, embedding, ts);
            if let Some(meta) = metadata {
                mem.metadata = meta;
            }
            mem
        } else {
            let mut mem = Memory::new(content, embedding);
            if let Some(meta) = metadata {
                mem.metadata = meta;
            }
            mem
        };

        // Add source if provided
        if let Some(src) = source {
            memory.set_source(src)?;
        }

        // Set namespace (defaults to empty string)
        memory.set_namespace(effective_ns.unwrap_or(""));

        // Delegate to ingestion pipeline for atomic indexing
        self.pipeline.add(memory)
    }

    /// Retrieve a memory by ID
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to retrieve
    ///
    /// # Returns
    ///
    /// The memory record if found, or None
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("test".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// let memory = engine.get(&id).unwrap();
    /// if let Some(mem) = memory {
    ///     println!("Content: {}", mem.content);
    /// }
    /// ```
    pub fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        self.storage.get_memory(id)
    }

    /// Delete a memory by ID
    ///
    /// This will remove the memory from all indexes.
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to delete
    /// * `namespace` - Optional namespace. If provided, verifies the memory is in this namespace before deleting
    ///
    /// # Returns
    ///
    /// true if the memory was deleted, false if it didn't exist
    ///
    /// # Errors
    ///
    /// Returns `Error::NamespaceMismatch` if namespace is provided and doesn't match
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("test".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// let deleted = engine.delete(&id, None).unwrap();
    /// assert!(deleted);
    /// ```
    pub fn delete(&self, id: &MemoryId, namespace: Option<&str>) -> Result<bool> {
        // If namespace is provided, verify it matches before deleting
        if let Some(expected_ns) = namespace {
            if let Some(memory) = self.storage.get_memory(id)? {
                let found_ns = memory.get_namespace();
                if found_ns != expected_ns {
                    return Err(Error::NamespaceMismatch {
                        expected: expected_ns.to_string(),
                        found: found_ns,
                    });
                }
            } else {
                // Memory doesn't exist
                return Ok(false);
            }
        }

        // Delegate to ingestion pipeline for atomic cleanup
        self.pipeline.delete(id)
    }

    /// Add multiple memories in a batch operation
    ///
    /// This is significantly faster than calling `add()` multiple times (10x+ improvement)
    /// because it uses:
    /// - Single transaction for all storage operations
    /// - Vector index locked once for all additions
    /// - Batched entity extraction with deduplication
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of MemoryInput to add
    ///
    /// # Returns
    ///
    /// BatchResult containing IDs of created memories and any errors
    ///
    /// # Performance
    ///
    /// Target: 1,000 memories in <500ms
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    /// use mnemefusion_core::types::MemoryInput;
    ///
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let inputs = vec![
    ///     MemoryInput::new("content 1".to_string(), vec![0.1; 384]),
    ///     MemoryInput::new("content 2".to_string(), vec![0.2; 384]),
    /// ];
    ///
    /// let result = engine.add_batch(inputs, None).unwrap();
    /// println!("Created {} memories", result.created_count);
    /// if result.has_errors() {
    ///     println!("Encountered {} errors", result.errors.len());
    /// }
    /// ```
    pub fn add_batch(
        &self,
        inputs: Vec<MemoryInput>,
        namespace: Option<&str>,
    ) -> Result<BatchResult> {
        // Validate all embeddings upfront
        for (index, input) in inputs.iter().enumerate() {
            if input.embedding.len() != self.config.embedding_dim {
                let mut result = BatchResult::new();
                result.errors.push(crate::types::BatchError::new(
                    index,
                    format!(
                        "Invalid embedding dimension: expected {}, got {}",
                        self.config.embedding_dim,
                        input.embedding.len()
                    ),
                ));
                return Ok(result);
            }
        }

        // Set namespace on all inputs if provided
        let mut inputs_with_ns = inputs;
        if let Some(ns) = namespace {
            for input in &mut inputs_with_ns {
                input.namespace = Some(ns.to_string());
            }
        }

        // Delegate to ingestion pipeline
        self.pipeline.add_batch(inputs_with_ns, None)
    }

    /// Delete multiple memories in a batch operation
    ///
    /// This is faster than calling `delete()` multiple times because it uses:
    /// - Single transaction for all storage operations
    /// - Batched entity cleanup
    ///
    /// # Arguments
    ///
    /// * `ids` - Vector of MemoryIds to delete
    /// * `namespace` - Optional namespace. If provided, only deletes memories in this namespace
    ///
    /// # Returns
    ///
    /// Number of memories actually deleted (may be less than input if some don't exist or are in wrong namespace)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    ///
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id1 = engine.add("test1".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// # let id2 = engine.add("test2".to_string(), vec![0.2; 384], None, None, None, None).unwrap();
    /// let ids = vec![id1, id2];
    /// let deleted_count = engine.delete_batch(ids, None).unwrap();
    /// println!("Deleted {} memories", deleted_count);
    /// ```
    pub fn delete_batch(&self, ids: Vec<MemoryId>, namespace: Option<&str>) -> Result<usize> {
        // If namespace is provided, filter IDs to only those in the namespace
        let ids_to_delete = if let Some(expected_ns) = namespace {
            let mut filtered_ids = Vec::new();
            for id in ids {
                if let Some(memory) = self.storage.get_memory(&id)? {
                    if memory.get_namespace() == expected_ns {
                        filtered_ids.push(id);
                    }
                }
            }
            filtered_ids
        } else {
            ids
        };

        // Delegate to ingestion pipeline
        self.pipeline.delete_batch(ids_to_delete)
    }

    /// Add a memory with automatic deduplication
    ///
    /// Uses content hash to detect duplicates. If identical content already exists,
    /// returns the existing memory ID without creating a duplicate.
    ///
    /// # Arguments
    ///
    /// * `content` - Text content
    /// * `embedding` - Vector embedding
    /// * `metadata` - Optional metadata
    /// * `timestamp` - Optional custom timestamp
    /// * `source` - Optional source/provenance
    ///
    /// # Returns
    ///
    /// AddResult with created flag and ID (either new or existing)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    ///
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let embedding = vec![0.1; 384];
    ///
    /// // First add
    /// let result1 = engine.add_with_dedup(
    ///     "Meeting notes".to_string(),
    ///     embedding.clone(),
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// assert!(result1.created);
    ///
    /// // Second add with same content
    /// let result2 = engine.add_with_dedup(
    ///     "Meeting notes".to_string(),
    ///     embedding.clone(),
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// assert!(!result2.created); // Duplicate detected
    /// assert_eq!(result1.id, result2.id); // Same ID returned
    /// ```
    pub fn add_with_dedup(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
        namespace: Option<&str>,
    ) -> Result<AddResult> {
        // Validate embedding dimension
        if embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }

        // Create memory
        let mut memory = if let Some(ts) = timestamp {
            Memory::new_with_timestamp(content, embedding, ts)
        } else {
            Memory::new(content, embedding)
        };

        // Add metadata
        if let Some(meta) = metadata {
            for (key, value) in meta {
                memory.set_metadata(key, value);
            }
        }

        // Add source
        if let Some(src) = source {
            memory.set_source(src)?;
        }

        // Set namespace if provided
        memory.set_namespace(namespace.unwrap_or(""));

        // Delegate to pipeline with deduplication
        self.pipeline.add_with_dedup(memory)
    }

    /// Upsert a memory by logical key
    ///
    /// If key exists: replaces content, embedding, and metadata
    /// If key doesn't exist: creates new memory and associates with key
    ///
    /// This is useful for updating facts that may change over time.
    ///
    /// # Arguments
    ///
    /// * `key` - Logical key (e.g., "user_profile:123", "doc:readme")
    /// * `content` - Text content
    /// * `embedding` - Vector embedding
    /// * `metadata` - Optional metadata
    /// * `timestamp` - Optional custom timestamp
    /// * `source` - Optional source/provenance
    ///
    /// # Returns
    ///
    /// UpsertResult indicating whether memory was created or updated
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    ///
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let embedding = vec![0.1; 384];
    ///
    /// // First upsert - creates new
    /// let result1 = engine.upsert(
    ///     "user:profile",
    ///     "Alice likes hiking".to_string(),
    ///     embedding.clone(),
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// assert!(result1.created);
    ///
    /// // Second upsert - updates existing
    /// let result2 = engine.upsert(
    ///     "user:profile",
    ///     "Alice likes hiking and photography".to_string(),
    ///     vec![0.2; 384],
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// assert!(result2.updated);
    /// assert_eq!(result2.previous_content, Some("Alice likes hiking".to_string()));
    /// ```
    pub fn upsert(
        &self,
        key: &str,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
        namespace: Option<&str>,
    ) -> Result<UpsertResult> {
        // Validate embedding dimension
        if embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }

        // Create memory
        let mut memory = if let Some(ts) = timestamp {
            Memory::new_with_timestamp(content, embedding, ts)
        } else {
            Memory::new(content, embedding)
        };

        // Add metadata
        if let Some(meta) = metadata {
            for (key, value) in meta {
                memory.set_metadata(key, value);
            }
        }

        // Add source
        if let Some(src) = source {
            memory.set_source(src)?;
        }

        // Set namespace if provided
        memory.set_namespace(namespace.unwrap_or(""));

        // Delegate to pipeline
        self.pipeline.upsert(key, memory)
    }

    /// Get the number of memories in the database
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let count = engine.count().unwrap();
    /// println!("Total memories: {}", count);
    /// ```
    pub fn count(&self) -> Result<usize> {
        self.storage.count_memories()
    }

    /// List all memory IDs (for debugging/testing)
    ///
    /// # Warning
    ///
    /// This loads all memory IDs into memory. Use with caution on large databases.
    pub fn list_ids(&self) -> Result<Vec<MemoryId>> {
        self.storage.list_memory_ids()
    }

    /// Update the embedding vector for an existing memory.
    ///
    /// This updates both the stored memory record (used by MMR diversity) and
    /// the HNSW vector index (used by semantic search). The memory content,
    /// metadata, and all other fields are preserved.
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to update
    /// * `new_embedding` - The new embedding vector (must match configured dimension)
    ///
    /// # Errors
    ///
    /// Returns error if the memory doesn't exist or the embedding dimension is wrong.
    pub fn update_embedding(&self, id: &MemoryId, new_embedding: Vec<f32>) -> Result<()> {
        // Load existing memory
        let mut memory = self
            .storage
            .get_memory(id)?
            .ok_or_else(|| Error::MemoryNotFound(id.to_string()))?;

        // Validate dimension
        let expected = self.config.embedding_dim;
        if new_embedding.len() != expected {
            return Err(Error::InvalidEmbeddingDimension {
                expected,
                got: new_embedding.len(),
            });
        }

        // Update embedding in memory record
        memory.embedding = new_embedding.clone();

        // Update storage (redb)
        self.storage.store_memory(&memory)?;

        // Update vector index (usearch HNSW)
        let mut vi = self.vector_index.write().unwrap();
        // Remove old vector, then add new one
        let _ = vi.remove(id); // ignore error if not found (fresh index)
        vi.add(id.clone(), &new_embedding)?;

        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Reserve capacity in the vector index for future insertions
    ///
    /// This is useful when you know you'll be adding many memories
    /// and want to avoid repeated reallocations, improving performance.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of vectors to reserve space for
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let mut engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// // Reserve space for 10,000 memories before bulk insertion
    /// engine.reserve_capacity(10_000).unwrap();
    /// ```
    pub fn reserve_capacity(&self, capacity: usize) -> Result<()> {
        self.pipeline.reserve_capacity(capacity)
    }

    /// Search for memories by semantic similarity
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query vector to search for
    /// * `top_k` - Maximum number of results to return
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
    ///
    /// # Returns
    ///
    /// A vector of (Memory, similarity_score) tuples, sorted by similarity (highest first)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let query_embedding = vec![0.1; 384];
    /// let results = engine.search(&query_embedding, 10, None, None).unwrap();
    /// for (memory, score) in results {
    ///     println!("Similarity: {:.3} - {}", score, memory.content);
    /// }
    /// ```
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        namespace: Option<&str>,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<Vec<(Memory, f32)>> {
        // If filtering is needed, fetch more results (5x) and filter
        let needs_filtering =
            namespace.is_some() || (filters.is_some() && !filters.unwrap().is_empty());
        let fetch_k = if needs_filtering { top_k * 5 } else { top_k };

        // Search vector index
        let vector_results = {
            let index = self.vector_index.read().unwrap();
            index.search(query_embedding, fetch_k)?
        };

        // Retrieve full memory records using u64 lookup
        let mut results = Vec::with_capacity(vector_results.len());

        for vector_result in vector_results {
            // Look up memory using the u64 key from vector index
            let key = vector_result.id.to_u64();
            if let Some(memory) = self.storage.get_memory_by_u64(key)? {
                // Filter by namespace if provided
                if let Some(ns) = namespace {
                    if memory.get_namespace() != ns {
                        continue;
                    }
                }

                // Filter by metadata if provided
                if let Some(filter_list) = filters {
                    if !Self::memory_matches_filters(&memory, filter_list) {
                        continue;
                    }
                }

                results.push((memory, vector_result.similarity));

                // Stop if we have enough results after filtering
                if results.len() >= top_k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Check if a memory matches all metadata filters
    fn memory_matches_filters(memory: &Memory, filters: &[MetadataFilter]) -> bool {
        for filter in filters {
            let value = memory.metadata.get(&filter.field).map(|s| s.as_str());
            if !filter.matches(value) {
                return false;
            }
        }
        true
    }

    /// Intelligent multi-dimensional query with intent classification
    ///
    /// This method performs intent-aware retrieval across all dimensions:
    /// - Classifies the query intent (temporal, causal, entity, factual)
    /// - Retrieves results from relevant dimensions
    /// - Fuses results with adaptive weights based on intent
    ///
    /// # Arguments
    ///
    /// * `query_text` - Natural language query text
    /// * `query_embedding` - Vector embedding of the query
    /// * `limit` - Maximum number of results to return
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
    ///
    /// # Returns
    ///
    /// Tuple of (intent classification, fused results with full memory records)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("test.mfdb", Config::default()).unwrap();
    /// # let query_embedding = vec![0.1; 384];
    /// let (intent, results, profile_context) = engine.query(
    ///     "Why was the meeting cancelled?",
    ///     &query_embedding,
    ///     10,
    ///     None,
    ///     None
    /// ).unwrap();
    ///
    /// println!("Query intent: {:?}", intent.intent);
    /// println!("Profile context: {} entries", profile_context.len());
    /// for result in results {
    ///     println!("Score: {:.3} - {}", result.1.fused_score, result.0.content);
    /// }
    /// ```
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: impl Into<Option<Vec<f32>>>,
        limit: usize,
        namespace: Option<&str>,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<(IntentClassification, Vec<(Memory, FusedResult)>, Vec<String>)> {
        // Resolve query embedding: use provided value or auto-compute from query text
        let embedding_vec: Vec<f32> = match query_embedding.into() {
            Some(e) => e,
            None => self.auto_embed(query_text)?,
        };

        // Apply default namespace if caller didn't supply one
        let effective_ns = namespace.or(self.default_namespace.as_deref());

        // Execute query using query planner
        let (intent, fused_results, matched_facts) =
            self.query_planner
                .query(query_text, &embedding_vec, limit, effective_ns, filters)?;

        // Build profile context as SEPARATE strings (not mixed into results).
        // Profile facts contain entity knowledge ("Caroline's hobby: painting") but
        // lack dates, speaker context, and conversational detail. Mixing them into
        // the results Vec with high scores pushes real memories out of top-K context.
        let mut profile_context = Vec::new();

        // Group matched facts by entity name
        let mut facts_by_entity: HashMap<String, Vec<&crate::query::MatchedProfileFact>> = HashMap::new();
        for fact in &matched_facts {
            facts_by_entity
                .entry(fact.entity_name.clone())
                .or_default()
                .push(fact);
        }

        for (entity_name, facts) in &facts_by_entity {
            // If profile has a pre-computed summary, use it as ONE context item
            // Otherwise, fall back to individual fact format
            let profile_summary = self
                .storage
                .get_entity_profile(entity_name)
                .ok()
                .flatten()
                .and_then(|p| p.summary.clone());

            if let Some(summary) = profile_summary {
                profile_context.push(summary);
            } else {
                // No summary — format individual facts
                for fact in facts {
                    let content = format!(
                        "{}'s {}: {}",
                        fact.entity_name,
                        fact.fact_type.replace('_', " "),
                        fact.value
                    );
                    profile_context.push(content);
                }
            }
        }

        // Retrieve full memory records using u64 key lookup
        // Note: Vector index returns partial MemoryIds (first 8 bytes only),
        // so we use get_memory_by_u64 which looks up the full UUID from the index table
        let mut results = Vec::new();
        for fused_result in fused_results {
            let key = fused_result.id.to_u64();
            if let Some(memory) = self.storage.get_memory_by_u64(key)? {
                results.push((memory, fused_result));
            }
        }

        Ok((intent, results, profile_context))
    }

    /// Query memories within a time range
    ///
    /// Returns memories whose timestamps fall within the specified range,
    /// sorted by timestamp (newest first).
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the time range (inclusive)
    /// * `end` - End of the time range (inclusive)
    /// * `limit` - Maximum number of results to return
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
    ///
    /// # Returns
    ///
    /// A vector of (Memory, Timestamp) tuples, sorted newest first
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config, Timestamp};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let now = Timestamp::now();
    /// let week_ago = now.subtract_days(7);
    ///
    /// let results = engine.get_range(week_ago, now, 100, None).unwrap();
    /// for (memory, timestamp) in results {
    ///     println!("{}: {}", timestamp.as_unix_secs(), memory.content);
    /// }
    /// ```
    pub fn get_range(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(Memory, Timestamp)>> {
        // Fetch more results if filtering by namespace
        let fetch_limit = if namespace.is_some() {
            limit * 3
        } else {
            limit
        };
        let temporal_results = self.temporal_index.range_query(start, end, fetch_limit)?;

        // Retrieve and filter full memory records
        let mut results = Vec::with_capacity(temporal_results.len());

        for temp_result in temporal_results {
            if let Some(memory) = self.storage.get_memory(&temp_result.id)? {
                // Filter by namespace if provided
                if let Some(ns) = namespace {
                    if memory.get_namespace() != ns {
                        continue;
                    }
                }
                results.push((memory, temp_result.timestamp));

                // Stop if we have enough results after filtering
                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Get the N most recent memories
    ///
    /// Returns the most recent memories, sorted by timestamp (newest first).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of recent memories to retrieve
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
    ///
    /// # Returns
    ///
    /// A vector of (Memory, Timestamp) tuples, sorted newest first
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let recent = engine.get_recent(10, None).unwrap();
    /// println!("10 most recent memories:");
    /// for (memory, timestamp) in recent {
    ///     println!("  {} - {}", timestamp.as_unix_secs(), memory.content);
    /// }
    /// ```
    pub fn get_recent(
        &self,
        n: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(Memory, Timestamp)>> {
        // Fetch more results if filtering by namespace
        let fetch_n = if namespace.is_some() { n * 3 } else { n };
        let temporal_results = self.temporal_index.recent(fetch_n)?;

        // Retrieve and filter full memory records
        let mut results = Vec::with_capacity(temporal_results.len());

        for temp_result in temporal_results {
            if let Some(memory) = self.storage.get_memory(&temp_result.id)? {
                // Filter by namespace if provided
                if let Some(ns) = namespace {
                    if memory.get_namespace() != ns {
                        continue;
                    }
                }
                results.push((memory, temp_result.timestamp));

                // Stop if we have enough results after filtering
                if results.len() >= n {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Add a causal link between two memories
    ///
    /// Links a cause memory to an effect memory with a confidence score.
    ///
    /// # Arguments
    ///
    /// * `cause` - The MemoryId of the cause
    /// * `effect` - The MemoryId of the effect
    /// * `confidence` - Confidence score (0.0 to 1.0)
    /// * `evidence` - Evidence text explaining the causal relationship
    ///
    /// # Errors
    ///
    /// Returns error if confidence is not in range [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id1 = engine.add("Cause".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// # let id2 = engine.add("Effect".to_string(), vec![0.2; 384], None, None, None, None).unwrap();
    /// engine.add_causal_link(&id1, &id2, 0.9, "id1 caused id2".to_string()).unwrap();
    /// ```
    pub fn add_causal_link(
        &self,
        cause: &MemoryId,
        effect: &MemoryId,
        confidence: f32,
        evidence: String,
    ) -> Result<()> {
        // Add the causal link to the graph
        {
            let mut graph = self.graph_manager.write().unwrap();
            graph.add_causal_link(cause, effect, confidence, evidence)?;
        }

        // Persist graph immediately for crash recovery
        // This ensures causal links are durable
        {
            let graph = self.graph_manager.read().unwrap();
            crate::graph::persist::save_graph(&graph, &self.storage)?;
        }

        Ok(())
    }

    /// Get causes of a memory (backward traversal)
    ///
    /// Finds all memories that causally precede the given memory, up to max_hops.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory to find causes for
    /// * `max_hops` - Maximum traversal depth
    ///
    /// # Returns
    ///
    /// CausalTraversalResult with all paths found
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("Memory".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// let causes = engine.get_causes(&id, 3).unwrap();
    /// for path in causes.paths {
    ///     println!("Found causal path with {} steps (confidence: {})",
    ///              path.memories.len(), path.confidence);
    /// }
    /// ```
    pub fn get_causes(
        &self,
        memory_id: &MemoryId,
        max_hops: usize,
    ) -> Result<CausalTraversalResult> {
        let graph = self.graph_manager.read().unwrap();
        graph.get_causes(memory_id, max_hops)
    }

    /// Get effects of a memory (forward traversal)
    ///
    /// Finds all memories that causally follow the given memory, up to max_hops.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory to find effects for
    /// * `max_hops` - Maximum traversal depth
    ///
    /// # Returns
    ///
    /// CausalTraversalResult with all paths found
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("Memory".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// let effects = engine.get_effects(&id, 3).unwrap();
    /// for path in effects.paths {
    ///     println!("Found effect chain with {} steps (confidence: {})",
    ///              path.memories.len(), path.confidence);
    /// }
    /// ```
    pub fn get_effects(
        &self,
        memory_id: &MemoryId,
        max_hops: usize,
    ) -> Result<CausalTraversalResult> {
        let graph = self.graph_manager.read().unwrap();
        graph.get_effects(memory_id, max_hops)
    }

    // ========== Namespace Operations ==========

    /// List all namespaces in the database
    ///
    /// Returns a sorted list of all unique namespace strings, excluding the default namespace ("").
    ///
    /// # Performance
    ///
    /// O(n) where n = total memories. This scans all memories to extract namespaces.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let namespaces = engine.list_namespaces().unwrap();
    /// for ns in namespaces {
    ///     println!("Namespace: {}", ns);
    /// }
    /// ```
    pub fn list_namespaces(&self) -> Result<Vec<String>> {
        self.storage.list_namespaces()
    }

    /// Count memories in a specific namespace
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace to count (empty string "" for default namespace)
    ///
    /// # Returns
    ///
    /// Number of memories in the namespace
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let count = engine.count_namespace("user_123").unwrap();
    /// println!("User has {} memories", count);
    /// ```
    pub fn count_namespace(&self, namespace: &str) -> Result<usize> {
        self.storage.count_namespace(namespace)
    }

    /// Delete all memories in a namespace
    ///
    /// This is a convenience method that lists all memory IDs in the namespace
    /// and deletes them via the ingestion pipeline (ensuring proper cleanup of indexes).
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace to delete (empty string "" for default namespace)
    ///
    /// # Returns
    ///
    /// Number of memories deleted
    ///
    /// # Warning
    ///
    /// This operation cannot be undone. Use with caution.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let deleted = engine.delete_namespace("old_user").unwrap();
    /// println!("Deleted {} memories from namespace", deleted);
    /// ```
    pub fn delete_namespace(&self, namespace: &str) -> Result<usize> {
        // Get all memory IDs in this namespace
        let ids = self.storage.list_namespace_ids(namespace)?;

        // Delete via pipeline for proper cleanup
        self.pipeline.delete_batch(ids)
    }

    // ========== Entity Operations ==========

    /// Get all memories that mention a specific entity
    ///
    /// # Arguments
    ///
    /// * `entity_name` - The name of the entity to query (case-insensitive)
    ///
    /// # Returns
    ///
    /// A vector of Memory objects that mention this entity
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let memories = engine.get_entity_memories("Project Alpha").unwrap();
    /// for memory in memories {
    ///     println!("{}", memory.content);
    /// }
    /// ```
    pub fn get_entity_memories(&self, entity_name: &str) -> Result<Vec<Memory>> {
        // Find the entity by name
        let entity = self.storage.find_entity_by_name(entity_name)?;

        match entity {
            Some(entity) => {
                // Query entity graph
                let graph = self.graph_manager.read().unwrap();
                let result = graph.get_entity_memories(&entity.id);

                // Retrieve full memory records
                let mut memories = Vec::with_capacity(result.memories.len());
                for memory_id in result.memories {
                    if let Some(memory) = self.storage.get_memory(&memory_id)? {
                        memories.push(memory);
                    }
                }

                Ok(memories)
            }
            None => Ok(Vec::new()), // Entity not found, return empty list
        }
    }

    /// Get all entities mentioned in a specific memory
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory to query
    ///
    /// # Returns
    ///
    /// A vector of Entity objects mentioned in this memory
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("Alice met Bob".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// let entities = engine.get_memory_entities(&id).unwrap();
    /// for entity in entities {
    ///     println!("Entity: {}", entity.name);
    /// }
    /// ```
    pub fn get_memory_entities(&self, memory_id: &MemoryId) -> Result<Vec<Entity>> {
        // Query entity graph
        let graph = self.graph_manager.read().unwrap();
        let entity_ids = graph.get_memory_entities(memory_id);

        // Retrieve full entity records
        let mut entities = Vec::with_capacity(entity_ids.len());
        for entity_id in entity_ids {
            if let Some(entity) = self.storage.get_entity(&entity_id)? {
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    /// List all entities in the database
    ///
    /// # Returns
    ///
    /// A vector of all Entity objects
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let all_entities = engine.list_entities().unwrap();
    /// for entity in all_entities {
    ///     println!("{}: {} mentions", entity.name, entity.mention_count);
    /// }
    /// ```
    pub fn list_entities(&self) -> Result<Vec<Entity>> {
        self.storage.list_entities()
    }

    // ========== Entity Profile Operations ==========

    /// Get the profile for an entity by name
    ///
    /// Entity profiles aggregate facts about entities across all memories.
    /// They are automatically built during ingestion when SLM metadata extraction
    /// is enabled.
    ///
    /// # Arguments
    ///
    /// * `name` - The entity name (case-insensitive)
    ///
    /// # Returns
    ///
    /// The EntityProfile if found, or None
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// if let Some(profile) = engine.get_entity_profile("Alice").unwrap() {
    ///     println!("Entity: {} ({})", profile.name, profile.entity_type);
    ///
    ///     // Get facts about Alice's occupation
    ///     for fact in profile.get_facts("occupation") {
    ///         println!("  Occupation: {} (confidence: {})", fact.value, fact.confidence);
    ///     }
    ///
    ///     // Get facts about Alice's research
    ///     for fact in profile.get_facts("research_topic") {
    ///         println!("  Research: {} (confidence: {})", fact.value, fact.confidence);
    ///     }
    /// }
    /// ```
    pub fn get_entity_profile(&self, name: &str) -> Result<Option<EntityProfile>> {
        self.storage.get_entity_profile(name)
    }

    /// List all entity profiles in the database
    ///
    /// # Returns
    ///
    /// A vector of all EntityProfile objects
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let profiles = engine.list_entity_profiles().unwrap();
    /// for profile in profiles {
    ///     println!("{} ({}) - {} facts from {} memories",
    ///         profile.name,
    ///         profile.entity_type,
    ///         profile.total_facts(),
    ///         profile.source_memories.len()
    ///     );
    /// }
    /// ```
    pub fn list_entity_profiles(&self) -> Result<Vec<EntityProfile>> {
        self.storage.list_entity_profiles()
    }

    /// Count entity profiles in the database
    ///
    /// # Returns
    ///
    /// The number of entity profiles
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let count = engine.count_entity_profiles().unwrap();
    /// println!("Total entity profiles: {}", count);
    /// ```
    pub fn count_entity_profiles(&self) -> Result<usize> {
        self.storage.count_entity_profiles()
    }

    /// Create a scoped view for namespace-specific operations
    ///
    /// Returns a ScopedMemory that automatically applies the namespace to all operations.
    /// This provides a more ergonomic API when working with a single namespace.
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace to scope to (empty string "" for default namespace)
    ///
    /// # Returns
    ///
    /// A ScopedMemory view bound to this namespace
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// // Create scoped view for a user
    /// let user_memory = engine.scope("user_123");
    ///
    /// // All operations automatically use the namespace
    /// let id = user_memory.add("User note".to_string(), vec![0.1; 384], None, None, None).unwrap();
    /// let results = user_memory.search(&vec![0.1; 384], 10, None).unwrap();
    /// let count = user_memory.count().unwrap();
    /// user_memory.delete_all().unwrap();
    /// ```
    pub fn scope<S: Into<String>>(&self, namespace: S) -> ScopedMemory {
        ScopedMemory {
            engine: self,
            namespace: namespace.into(),
        }
    }

    /// Close the database
    ///
    /// This saves all indexes and ensures all data is flushed to disk.
    /// While not strictly necessary (redb handles persistence automatically),
    /// it's good practice to call this explicitly when you're done.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// // ... use engine ...
    /// engine.close().unwrap();
    /// ```
    pub fn close(self) -> Result<()> {
        // Save vector index
        {
            let index = self.vector_index.read().unwrap();
            index.save()?;
        }

        // Save BM25 index
        self.bm25_index.save()?;

        // Save causal graph
        {
            let graph = self.graph_manager.read().unwrap();
            crate::graph::persist::save_graph(&graph, &self.storage)?;
        }

        // Storage (redb) handles persistence automatically
        Ok(())
    }
}

/// Scoped memory view for namespace-specific operations
///
/// This wrapper provides an ergonomic API for working with a single namespace.
/// All operations automatically apply the namespace, eliminating the need to pass
/// it to every method call.
///
/// Created via `MemoryEngine::scope()`.
///
/// # Example
///
/// ```no_run
/// use mnemefusion_core::{MemoryEngine, Config};
///
/// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
/// // Create scoped view for a specific user
/// let user_memory = engine.scope("user_123");
///
/// // Add memory (automatically in user_123 namespace)
/// let id = user_memory.add(
///     "User note".to_string(),
///     vec![0.1; 384],
///     None,
///     None,
///     None,
/// ).unwrap();
///
/// // Search within namespace
/// let results = user_memory.search(&vec![0.1; 384], 10, None).unwrap();
///
/// // Count memories in namespace
/// println!("User has {} memories", user_memory.count().unwrap());
///
/// // Delete all memories in namespace
/// user_memory.delete_all().unwrap();
/// ```
pub struct ScopedMemory<'a> {
    engine: &'a MemoryEngine,
    namespace: String,
}

impl<'a> ScopedMemory<'a> {
    /// Add a memory to this namespace
    ///
    /// Equivalent to calling `engine.add(..., Some(namespace))`
    ///
    /// # Arguments
    ///
    /// * `content` - Text content
    /// * `embedding` - Vector embedding
    /// * `metadata` - Optional metadata
    /// * `timestamp` - Optional custom timestamp
    /// * `source` - Optional source/provenance
    ///
    /// # Returns
    ///
    /// The ID of the created memory
    pub fn add(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
    ) -> Result<MemoryId> {
        self.engine.add(
            content,
            embedding,
            metadata,
            timestamp,
            source,
            Some(&self.namespace),
        )
    }

    /// Search for memories in this namespace
    ///
    /// Equivalent to calling `engine.search(..., Some(namespace), filters)`
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - Query vector
    /// * `top_k` - Maximum number of results
    /// * `filters` - Optional metadata filters
    ///
    /// # Returns
    ///
    /// Vector of (Memory, similarity_score) tuples
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<Vec<(Memory, f32)>> {
        self.engine
            .search(query_embedding, top_k, Some(&self.namespace), filters)
    }

    /// Delete a memory from this namespace
    ///
    /// Equivalent to calling `engine.delete(..., Some(namespace))`
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to delete
    ///
    /// # Returns
    ///
    /// true if deleted, false if not found
    ///
    /// # Errors
    ///
    /// Returns `Error::NamespaceMismatch` if the memory exists but is in a different namespace
    pub fn delete(&self, id: &MemoryId) -> Result<bool> {
        self.engine.delete(id, Some(&self.namespace))
    }

    /// Add multiple memories to this namespace in a batch
    ///
    /// Equivalent to calling `engine.add_batch(..., Some(namespace))`
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of MemoryInput
    ///
    /// # Returns
    ///
    /// BatchResult with IDs and error information
    pub fn add_batch(&self, inputs: Vec<MemoryInput>) -> Result<BatchResult> {
        self.engine.add_batch(inputs, Some(&self.namespace))
    }

    /// Delete multiple memories from this namespace
    ///
    /// Equivalent to calling `engine.delete_batch(..., Some(namespace))`
    ///
    /// # Arguments
    ///
    /// * `ids` - Vector of memory IDs
    ///
    /// # Returns
    ///
    /// Number of memories deleted
    pub fn delete_batch(&self, ids: Vec<MemoryId>) -> Result<usize> {
        self.engine.delete_batch(ids, Some(&self.namespace))
    }

    /// Add a memory with deduplication in this namespace
    ///
    /// Equivalent to calling `engine.add_with_dedup(..., Some(namespace))`
    pub fn add_with_dedup(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
    ) -> Result<AddResult> {
        self.engine.add_with_dedup(
            content,
            embedding,
            metadata,
            timestamp,
            source,
            Some(&self.namespace),
        )
    }

    /// Upsert a memory in this namespace
    ///
    /// Equivalent to calling `engine.upsert(..., Some(namespace))`
    pub fn upsert(
        &self,
        key: &str,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
    ) -> Result<UpsertResult> {
        self.engine.upsert(
            key,
            content,
            embedding,
            metadata,
            timestamp,
            source,
            Some(&self.namespace),
        )
    }

    /// Count memories in this namespace
    ///
    /// Equivalent to calling `engine.count_namespace(namespace)`
    ///
    /// # Returns
    ///
    /// Number of memories in the namespace
    pub fn count(&self) -> Result<usize> {
        self.engine.count_namespace(&self.namespace)
    }

    /// Delete all memories in this namespace
    ///
    /// Equivalent to calling `engine.delete_namespace(namespace)`
    ///
    /// # Returns
    ///
    /// Number of memories deleted
    ///
    /// # Warning
    ///
    /// This operation cannot be undone. Use with caution.
    pub fn delete_all(&self) -> Result<usize> {
        self.engine.delete_namespace(&self.namespace)
    }

    /// Multi-dimensional query within this namespace
    ///
    /// Equivalent to calling `engine.query(..., Some(namespace), filters)`
    ///
    /// # Arguments
    ///
    /// * `query_text` - Natural language query
    /// * `query_embedding` - Query vector
    /// * `limit` - Maximum number of results
    /// * `filters` - Optional metadata filters
    ///
    /// # Returns
    ///
    /// Tuple of (intent classification, results, profile context)
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<(IntentClassification, Vec<(Memory, FusedResult)>, Vec<String>)> {
        self.engine.query(
            query_text,
            query_embedding.to_vec(),
            limit,
            Some(&self.namespace),
            filters,
        )
    }

    /// Get memories in time range within this namespace
    ///
    /// Equivalent to calling `engine.get_range(..., Some(namespace))`
    pub fn get_range(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
    ) -> Result<Vec<(Memory, Timestamp)>> {
        self.engine
            .get_range(start, end, limit, Some(&self.namespace))
    }

    /// Get recent memories within this namespace
    ///
    /// Equivalent to calling `engine.get_recent(..., Some(namespace))`
    pub fn get_recent(&self, n: usize) -> Result<Vec<(Memory, Timestamp)>> {
        self.engine.get_recent(n, Some(&self.namespace))
    }

    /// Get the namespace this view is scoped to
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_engine_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let engine = MemoryEngine::open(&path, Config::default()).unwrap();
        assert_eq!(engine.config().embedding_dim, 384);
    }

    #[test]
    fn test_memory_engine_invalid_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let mut config = Config::default();
        config.embedding_dim = 0;

        let result = MemoryEngine::open(&path, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_engine_add_and_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let content = "Test memory content".to_string();
        let embedding = vec![0.1; 384];

        let id = engine
            .add(content.clone(), embedding.clone(), None, None, None, None)
            .unwrap();

        let memory = engine.get(&id).unwrap();
        assert!(memory.is_some());

        let memory = memory.unwrap();
        assert_eq!(memory.content, content);
        assert_eq!(memory.embedding, embedding);
    }

    #[test]
    fn test_memory_engine_invalid_dimension() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let result = engine.add(
            "test".to_string(),
            vec![0.1; 512], // Wrong dimension
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_engine_with_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let id = engine
            .add(
                "test".to_string(),
                vec![0.1; 384],
                Some(metadata),
                None,
                None,
                None,
            )
            .unwrap();

        let memory = engine.get(&id).unwrap().unwrap();
        assert_eq!(memory.metadata.get("source"), Some(&"test".to_string()));
    }

    #[test]
    fn test_memory_engine_with_custom_timestamp() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let ts = Timestamp::from_unix_secs(1609459200.0); // 2021-01-01
        let id = engine
            .add(
                "test".to_string(),
                vec![0.1; 384],
                None,
                Some(ts),
                None,
                None,
            )
            .unwrap();

        let memory = engine.get(&id).unwrap().unwrap();
        assert_eq!(memory.created_at, ts);
    }

    #[test]
    fn test_memory_engine_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let id = engine
            .add("test".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();

        let deleted = engine.delete(&id, None).unwrap();
        assert!(deleted);

        let memory = engine.get(&id).unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_memory_engine_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        assert_eq!(engine.count().unwrap(), 0);

        engine
            .add("test1".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();
        assert_eq!(engine.count().unwrap(), 1);

        engine
            .add("test2".to_string(), vec![0.2; 384], None, None, None, None)
            .unwrap();
        assert_eq!(engine.count().unwrap(), 2);
    }

    #[test]
    fn test_memory_engine_list_ids() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let id1 = engine
            .add("test1".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();
        let id2 = engine
            .add("test2".to_string(), vec![0.2; 384], None, None, None, None)
            .unwrap();

        let ids = engine.list_ids().unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_memory_engine_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let id = {
            let engine = MemoryEngine::open(&path, Config::default()).unwrap();
            let id = engine
                .add(
                    "persistent".to_string(),
                    vec![0.5; 384],
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
            engine.close().unwrap();
            id
        };

        // Reopen and verify
        {
            let engine = MemoryEngine::open(&path, Config::default()).unwrap();
            let memory = engine.get(&id).unwrap();
            assert!(memory.is_some());
            assert_eq!(memory.unwrap().content, "persistent");
        }
    }

    #[test]
    fn test_namespace_add_and_filter() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memories to different namespaces
        let id1 = engine
            .add(
                "User 1 memory".to_string(),
                vec![0.1; 384],
                None,
                None,
                None,
                Some("user_1"),
            )
            .unwrap();

        let id2 = engine
            .add(
                "User 2 memory".to_string(),
                vec![0.2; 384],
                None,
                None,
                None,
                Some("user_2"),
            )
            .unwrap();

        let id3 = engine
            .add(
                "Default memory".to_string(),
                vec![0.3; 384],
                None,
                None,
                None,
                None,
            )
            .unwrap();

        // Verify memories are in correct namespaces
        let mem1 = engine.get(&id1).unwrap().unwrap();
        assert_eq!(mem1.get_namespace(), "user_1");

        let mem2 = engine.get(&id2).unwrap().unwrap();
        assert_eq!(mem2.get_namespace(), "user_2");

        let mem3 = engine.get(&id3).unwrap().unwrap();
        assert_eq!(mem3.get_namespace(), "");

        // Test search with namespace filtering
        let query_embedding = vec![0.15; 384];

        // Search in user_1 namespace
        let results = engine
            .search(&query_embedding, 10, Some("user_1"), None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id1);

        // Search in user_2 namespace
        let results = engine
            .search(&query_embedding, 10, Some("user_2"), None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id2);

        // Search in default namespace
        let results = engine.search(&query_embedding, 10, Some(""), None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id3);

        // Search without namespace filter (should get all)
        let results = engine.search(&query_embedding, 10, None, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_namespace_delete_with_verification() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memory to namespace
        let id = engine
            .add(
                "User memory".to_string(),
                vec![0.1; 384],
                None,
                None,
                None,
                Some("user_1"),
            )
            .unwrap();

        // Try to delete with wrong namespace - should fail
        let result = engine.delete(&id, Some("user_2"));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::Error::NamespaceMismatch { .. }
        ));

        // Verify memory still exists
        assert!(engine.get(&id).unwrap().is_some());

        // Delete with correct namespace - should succeed
        let deleted = engine.delete(&id, Some("user_1")).unwrap();
        assert!(deleted);

        // Verify memory is gone
        assert!(engine.get(&id).unwrap().is_none());
    }

    #[test]
    fn test_namespace_management_methods() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memories to different namespaces
        engine
            .add(
                "Memory 1".to_string(),
                vec![0.1; 384],
                None,
                None,
                None,
                Some("ns1"),
            )
            .unwrap();
        engine
            .add(
                "Memory 2".to_string(),
                vec![0.2; 384],
                None,
                None,
                None,
                Some("ns1"),
            )
            .unwrap();
        engine
            .add(
                "Memory 3".to_string(),
                vec![0.3; 384],
                None,
                None,
                None,
                Some("ns2"),
            )
            .unwrap();
        engine
            .add(
                "Memory 4".to_string(),
                vec![0.4; 384],
                None,
                None,
                None,
                None,
            )
            .unwrap();

        // List namespaces
        let namespaces = engine.list_namespaces().unwrap();
        assert_eq!(namespaces.len(), 2);
        assert!(namespaces.contains(&"ns1".to_string()));
        assert!(namespaces.contains(&"ns2".to_string()));

        // Count in namespace
        assert_eq!(engine.count_namespace("ns1").unwrap(), 2);
        assert_eq!(engine.count_namespace("ns2").unwrap(), 1);
        assert_eq!(engine.count_namespace("").unwrap(), 1); // Default namespace

        // Delete entire namespace
        let deleted = engine.delete_namespace("ns1").unwrap();
        assert_eq!(deleted, 2);

        // Verify namespace is gone
        assert_eq!(engine.count_namespace("ns1").unwrap(), 0);
        let namespaces = engine.list_namespaces().unwrap();
        assert_eq!(namespaces.len(), 1);
        assert!(namespaces.contains(&"ns2".to_string()));

        // Total count should be 2 now
        assert_eq!(engine.count().unwrap(), 2);
    }

    #[test]
    fn test_namespace_batch_operations() {
        use crate::types::MemoryInput;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Create batch inputs
        let inputs = vec![
            MemoryInput::new("Memory 1".to_string(), vec![0.1; 384]),
            MemoryInput::new("Memory 2".to_string(), vec![0.2; 384]),
            MemoryInput::new("Memory 3".to_string(), vec![0.3; 384]),
        ];

        // Add batch with namespace
        let result = engine.add_batch(inputs, Some("batch_ns")).unwrap();
        assert_eq!(result.created_count, 3);
        assert!(result.is_success());

        // Verify all are in the namespace
        assert_eq!(engine.count_namespace("batch_ns").unwrap(), 3);

        // Batch delete with namespace filter
        let deleted = engine
            .delete_batch(result.ids.clone(), Some("batch_ns"))
            .unwrap();
        assert_eq!(deleted, 3);

        // Verify namespace is empty
        assert_eq!(engine.count_namespace("batch_ns").unwrap(), 0);
    }

    // ========== ScopedMemory Tests ==========

    #[test]
    fn test_scoped_memory_add_and_search() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Create scoped view
        let scoped = engine.scope("user_123");

        // Add memory via scoped view
        let id = scoped
            .add(
                "Scoped memory".to_string(),
                vec![0.5; 384],
                None,
                None,
                None,
            )
            .unwrap();

        // Verify memory is in the namespace
        let memory = engine.get(&id).unwrap().unwrap();
        assert_eq!(memory.get_namespace(), "user_123");
        assert_eq!(memory.content, "Scoped memory");

        // Search via scoped view
        let results = scoped.search(&vec![0.5; 384], 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id);

        // Verify namespace isolation
        assert_eq!(scoped.namespace(), "user_123");
    }

    #[test]
    fn test_scoped_memory_count_and_delete_all() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let scoped = engine.scope("user_456");

        // Add multiple memories
        scoped
            .add("Memory 1".to_string(), vec![0.1; 384], None, None, None)
            .unwrap();
        scoped
            .add("Memory 2".to_string(), vec![0.2; 384], None, None, None)
            .unwrap();
        scoped
            .add("Memory 3".to_string(), vec![0.3; 384], None, None, None)
            .unwrap();

        // Count via scoped view
        assert_eq!(scoped.count().unwrap(), 3);

        // Total engine count should also be 3
        assert_eq!(engine.count().unwrap(), 3);

        // Delete all via scoped view
        let deleted = scoped.delete_all().unwrap();
        assert_eq!(deleted, 3);

        // Verify namespace is empty
        assert_eq!(scoped.count().unwrap(), 0);
        assert_eq!(engine.count().unwrap(), 0);
    }

    #[test]
    fn test_scoped_memory_isolation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Create two scoped views
        let scope1 = engine.scope("ns1");
        let scope2 = engine.scope("ns2");

        // Add memories to each namespace
        let id1 = scope1
            .add("NS1 memory".to_string(), vec![0.1; 384], None, None, None)
            .unwrap();
        let id2 = scope2
            .add("NS2 memory".to_string(), vec![0.2; 384], None, None, None)
            .unwrap();

        // Each scope should only see its own memories
        assert_eq!(scope1.count().unwrap(), 1);
        assert_eq!(scope2.count().unwrap(), 1);

        // Search should be isolated
        let results1 = scope1.search(&vec![0.1; 384], 10, None).unwrap();
        assert_eq!(results1.len(), 1);
        assert_eq!(results1[0].0.id, id1);

        let results2 = scope2.search(&vec![0.2; 384], 10, None).unwrap();
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].0.id, id2);

        // Delete from scope1 shouldn't affect scope2
        scope1.delete_all().unwrap();
        assert_eq!(scope1.count().unwrap(), 0);
        assert_eq!(scope2.count().unwrap(), 1);
    }

    #[test]
    fn test_scoped_memory_delete_with_verification() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let scope1 = engine.scope("ns1");
        let scope2 = engine.scope("ns2");

        // Add memory to ns1
        let id = scope1
            .add("NS1 memory".to_string(), vec![0.1; 384], None, None, None)
            .unwrap();

        // Try to delete from wrong namespace - should fail
        let result = scope2.delete(&id);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::Error::NamespaceMismatch { .. }
        ));

        // Verify memory still exists
        assert_eq!(scope1.count().unwrap(), 1);

        // Delete from correct namespace - should succeed
        let deleted = scope1.delete(&id).unwrap();
        assert!(deleted);
        assert_eq!(scope1.count().unwrap(), 0);
    }

    #[test]
    fn test_scoped_memory_batch_operations() {
        use crate::types::MemoryInput;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let scoped = engine.scope("batch_scope");

        // Add batch via scoped view
        let inputs = vec![
            MemoryInput::new("Batch 1".to_string(), vec![0.1; 384]),
            MemoryInput::new("Batch 2".to_string(), vec![0.2; 384]),
            MemoryInput::new("Batch 3".to_string(), vec![0.3; 384]),
        ];

        let result = scoped.add_batch(inputs).unwrap();
        assert_eq!(result.created_count, 3);
        assert!(result.is_success());

        // Verify count
        assert_eq!(scoped.count().unwrap(), 3);

        // Delete batch via scoped view
        let deleted = scoped.delete_batch(result.ids).unwrap();
        assert_eq!(deleted, 3);

        // Verify empty
        assert_eq!(scoped.count().unwrap(), 0);
    }

    #[test]
    fn test_search_with_metadata_filters() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memories with different metadata
        let mut mem1 = Memory::new("Event 1".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        mem1.metadata
            .insert("priority".to_string(), "high".to_string());
        engine
            .add(
                mem1.content.clone(),
                mem1.embedding.clone(),
                Some(mem1.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        let mut mem2 = Memory::new("Event 2".to_string(), vec![0.11; 384]);
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        mem2.metadata
            .insert("priority".to_string(), "low".to_string());
        engine
            .add(
                mem2.content.clone(),
                mem2.embedding.clone(),
                Some(mem2.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        let mut mem3 = Memory::new("Task 1".to_string(), vec![0.12; 384]);
        mem3.metadata.insert("type".to_string(), "task".to_string());
        mem3.metadata
            .insert("priority".to_string(), "high".to_string());
        engine
            .add(
                mem3.content.clone(),
                mem3.embedding.clone(),
                Some(mem3.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        // Search with filter: type=event
        let filters = vec![MetadataFilter::eq("type", "event")];
        let results = engine
            .search(&vec![0.1; 384], 10, None, Some(&filters))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|(m, _)| m.metadata.get("type").unwrap() == "event"));

        // Search with filter: priority=high
        let filters = vec![MetadataFilter::eq("priority", "high")];
        let results = engine
            .search(&vec![0.1; 384], 10, None, Some(&filters))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|(m, _)| m.metadata.get("priority").unwrap() == "high"));

        // Search with multiple filters: type=event AND priority=high
        let filters = vec![
            MetadataFilter::eq("type", "event"),
            MetadataFilter::eq("priority", "high"),
        ];
        let results = engine
            .search(&vec![0.1; 384], 10, None, Some(&filters))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.content, "Event 1");
    }

    #[test]
    fn test_query_with_metadata_filters() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        // Disable semantic threshold for test (simple test embeddings)
        let config = Config::default().with_fusion_semantic_threshold(0.0);
        let engine = MemoryEngine::open(&path, config).unwrap();

        // Add memories with different metadata
        let mut mem1 = Memory::new("Important meeting".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        mem1.metadata
            .insert("priority".to_string(), "high".to_string());
        engine
            .add(
                mem1.content.clone(),
                mem1.embedding.clone(),
                Some(mem1.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        let mut mem2 = Memory::new("Casual meeting".to_string(), vec![0.11; 384]);
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        mem2.metadata
            .insert("priority".to_string(), "low".to_string());
        engine
            .add(
                mem2.content.clone(),
                mem2.embedding.clone(),
                Some(mem2.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        // Query with filter
        let filters = vec![MetadataFilter::eq("priority", "high")];
        let (_intent, results, _profile_ctx) = engine
            .query("meeting", vec![0.1f32; 384], 10, None, Some(&filters))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.content, "Important meeting");
    }

    #[test]
    fn test_scoped_memory_with_filters() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let scoped = engine.scope("user_123");

        // Add memories with different metadata to the namespace
        let mut mem1 = Memory::new("Event 1".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        scoped
            .add(
                mem1.content.clone(),
                mem1.embedding.clone(),
                Some(mem1.metadata.clone()),
                None,
                None,
            )
            .unwrap();

        let mut mem2 = Memory::new("Task 1".to_string(), vec![0.11; 384]);
        mem2.metadata.insert("type".to_string(), "task".to_string());
        scoped
            .add(
                mem2.content.clone(),
                mem2.embedding.clone(),
                Some(mem2.metadata.clone()),
                None,
                None,
            )
            .unwrap();

        // Search with filter in scoped view
        let filters = vec![MetadataFilter::eq("type", "event")];
        let results = scoped.search(&vec![0.1; 384], 10, Some(&filters)).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.content, "Event 1");
    }

    #[test]
    fn test_consolidate_merges_aliases() {
        use crate::types::{EntityFact, EntityId, EntityProfile};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let mem1 = MemoryId::new();
        let mem2 = MemoryId::new();

        // Create orphan "mel" profile with 1 fact
        let mut mel_profile = EntityProfile::new(
            EntityId::new(),
            "mel".to_string(),
            "person".to_string(),
        );
        mel_profile.add_fact(EntityFact::new(
            "hobby",
            "hiking",
            0.9,
            mem1.clone(),
        ));
        mel_profile.add_source_memory(mem1.clone());
        engine.storage.store_entity_profile(&mel_profile).unwrap();

        // Create canonical "melanie" profile with 2 facts
        let mut melanie_profile = EntityProfile::new(
            EntityId::new(),
            "melanie".to_string(),
            "person".to_string(),
        );
        melanie_profile.add_fact(EntityFact::new(
            "instrument",
            "guitar",
            0.9,
            mem2.clone(),
        ));
        melanie_profile.add_fact(EntityFact::new(
            "occupation",
            "teacher",
            0.8,
            mem2.clone(),
        ));
        melanie_profile.add_source_memory(mem2.clone());
        engine.storage.store_entity_profile(&melanie_profile).unwrap();

        // Verify both exist before consolidation
        assert!(engine.storage.get_entity_profile("mel").unwrap().is_some());
        assert!(engine.storage.get_entity_profile("melanie").unwrap().is_some());

        let (_facts_removed, profiles_deleted) = engine.consolidate_profiles().unwrap();

        // "mel" should be merged into "melanie" and deleted
        assert!(profiles_deleted >= 1, "At least 1 profile should be deleted (mel)");
        assert!(
            engine.storage.get_entity_profile("mel").unwrap().is_none(),
            "mel profile should be deleted after merge"
        );

        // "melanie" should have all facts merged
        let melanie = engine
            .storage
            .get_entity_profile("melanie")
            .unwrap()
            .expect("melanie should still exist");
        assert!(
            melanie.total_facts() >= 3,
            "melanie should have merged facts (hiking + guitar + teacher), got {}",
            melanie.total_facts()
        );
        assert!(
            melanie.source_memories.contains(&mem1),
            "melanie should have mem1 from merged mel"
        );
        assert!(
            melanie.source_memories.contains(&mem2),
            "melanie should have mem2 from original melanie"
        );
    }

    // ======= Embedding-auto tests =======

    #[test]
    fn test_add_explicit_embedding_still_works() {
        // Existing callers that pass Vec<f32> directly should compile and work
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();
        let embedding = vec![0.1f32; 384];
        // Vec<f32> implements Into<Option<Vec<f32>>> — this must compile
        let id = engine
            .add("Alice loves hiking".to_string(), embedding, None, None, None, None)
            .unwrap();
        assert!(engine.get(&id).unwrap().is_some());
    }

    #[test]
    fn test_add_some_embedding_works() {
        // Some(Vec<f32>) should also compile
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();
        let id = engine
            .add(
                "Test content".to_string(),
                Some(vec![0.2f32; 384]),
                None, None, None, None,
            )
            .unwrap();
        assert!(engine.get(&id).unwrap().is_some());
    }

    #[test]
    fn test_add_none_embedding_without_engine_errors() {
        // None embedding without embedding engine configured → Error::NoEmbeddingEngine
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();
        let result = engine.add(
            "Test".to_string(),
            None::<Vec<f32>>,
            None, None, None, None,
        );
        assert!(
            result.is_err(),
            "Expected error when no embedding engine configured"
        );
        assert!(matches!(result.unwrap_err(), Error::NoEmbeddingEngine));
    }

    #[test]
    fn test_with_user_sets_default_namespace() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default())
            .unwrap()
            .with_user("alice");
        assert_eq!(engine.default_namespace.as_deref(), Some("alice"));

        // add() with no explicit namespace should use "alice"
        let id = engine
            .add("Alice's memory".to_string(), vec![0.1f32; 384], None, None, None, None)
            .unwrap();
        let mem = engine.get(&id).unwrap().unwrap();
        assert_eq!(mem.get_namespace(), "alice");
    }

    #[test]
    fn test_query_none_embedding_without_engine_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();
        let result = engine.query("test query", None::<Vec<f32>>, 10, None, None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::NoEmbeddingEngine));
    }
}
