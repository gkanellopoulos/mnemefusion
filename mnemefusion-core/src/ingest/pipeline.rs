//! Ingestion pipeline for coordinating memory storage across all dimensions
//!
//! The ingestion pipeline ensures that memories are indexed across all dimensions
//! atomically. If any indexing step fails, changes are rolled back to maintain
//! consistency.
//!
//! # SLM Metadata Extraction
//!
//! When the `slm` feature is enabled and an SLM extractor is configured, the pipeline
//! extracts rich metadata at ingestion time. This enables fast, accurate retrieval
//! without query-time SLM inference.

use crate::{
    error::Result,
    graph::GraphManager,
    index::{BM25Index, TemporalIndex, VectorIndex},
    ingest::{get_causal_extractor, get_temporal_extractor, EntityExtractor, SimpleEntityExtractor},
    memory::EmbeddingFn,
    storage::StorageEngine,
    types::{
        AddResult, BatchError, BatchResult, Entity, Memory,
        MemoryId, MemoryInput, UpsertResult,
    },
    util::hash,
};
use std::sync::{Arc, RwLock};

#[cfg(feature = "slm")]
use crate::ingest::{SlmMetadata, SlmMetadataExtractor};

#[cfg(feature = "entity-extraction")]
use crate::extraction::{ExtractionResult, LlmEntityExtractor};

#[cfg(any(feature = "entity-extraction", feature = "slm"))]
use crate::{
    query::profile_search::fact_embedding_key,
    types::{EntityFact, EntityId, EntityProfile, Timestamp},
};

#[cfg(any(feature = "entity-extraction", feature = "slm"))]
use std::sync::Mutex;

/// A memory whose LLM extraction has been deferred (async mode).
///
/// When `async_extraction_threshold > 0`, content >= the threshold is stored
/// immediately and pushed here. `flush_extraction_queue()` drains this list.
#[cfg(feature = "entity-extraction")]
struct PendingExtraction {
    memory_id: MemoryId,
    content: String,
    speaker: Option<String>,
    session_date: Option<String>,
}

/// Coordinates memory ingestion across all dimensions
///
/// The IngestionPipeline ensures that all dimension indexes are updated
/// atomically when adding or deleting memories. This prevents partial
/// state if any indexing operation fails.
///
/// # SLM Metadata Extraction
///
/// When the `slm` feature is enabled and an SLM extractor is configured via
/// [`with_slm_extractor`](Self::with_slm_extractor), the pipeline extracts rich
/// metadata (entities, temporal markers, causal relationships, topics) at ingestion
/// time. This metadata is stored in the memory and enables fast retrieval.
pub struct IngestionPipeline {
    storage: Arc<StorageEngine>,
    vector_index: Arc<RwLock<VectorIndex>>,
    bm25_index: Arc<BM25Index>,
    temporal_index: Arc<TemporalIndex>,
    graph_manager: Arc<RwLock<GraphManager>>,
    entity_extractor: SimpleEntityExtractor,
    entity_extraction_enabled: bool,
    /// SLM metadata extractor (optional, requires `slm` feature)
    #[cfg(feature = "slm")]
    slm_extractor: Option<Arc<Mutex<SlmMetadataExtractor>>>,
    /// Native LLM entity extractor (optional, requires `entity-extraction` feature)
    /// This takes precedence over the Python SLM extractor when both are available.
    #[cfg(feature = "entity-extraction")]
    llm_extractor: Option<Arc<Mutex<LlmEntityExtractor>>>,
    /// Tracks LLM extraction count for automatic GPU context reset.
    /// After GPU_CONTEXT_RESET_INTERVAL extractions, the GPU context is dropped
    /// and lazily recreated to prevent CUDA memory fragmentation.
    #[cfg(feature = "entity-extraction")]
    llm_extraction_count: std::sync::atomic::AtomicUsize,
    /// Embedding function for computing fact embeddings at ingestion time.
    /// When set, each extracted fact gets an embedding for semantic matching.
    embedding_fn: Option<EmbeddingFn>,
    /// Number of LLM extraction passes per document.
    /// Multiple passes with different parameters capture diverse facts.
    /// Default: 1 (single pass, backward compatible).
    #[cfg(feature = "entity-extraction")]
    extraction_passes: usize,
    /// Entity types allowed to have profiles (case-insensitive).
    /// Empty = allow all types. Non-empty = filter to listed types only.
    profile_entity_types: Vec<String>,
    /// Content size (bytes) threshold for deferred LLM extraction.
    /// 0 = always sync (default). Content >= threshold is deferred.
    async_extraction_threshold: usize,
    /// Queue of memories awaiting deferred LLM extraction.
    #[cfg(feature = "entity-extraction")]
    pending_extractions: Mutex<Vec<PendingExtraction>>,
}

impl IngestionPipeline {
    /// Number of LLM extractions between automatic GPU context resets.
    /// After this many inference cycles, the CUDA context is dropped and lazily
    /// recreated on the next call to prevent GPU memory fragmentation that can
    /// cause unrecoverable crashes in llama.cpp's C++ layer.
    #[cfg(feature = "entity-extraction")]
    const GPU_CONTEXT_RESET_INTERVAL: usize = 500;

    /// Create a new ingestion pipeline
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage engine for persistent data
    /// * `vector_index` - Vector index for semantic similarity
    /// * `bm25_index` - BM25 index for keyword search
    /// * `temporal_index` - Temporal index for time-based queries
    /// * `graph_manager` - Graph manager for causal and entity relationships
    /// * `entity_extraction_enabled` - Whether to automatically extract entities
    pub fn new(
        storage: Arc<StorageEngine>,
        vector_index: Arc<RwLock<VectorIndex>>,
        bm25_index: Arc<BM25Index>,
        temporal_index: Arc<TemporalIndex>,
        graph_manager: Arc<RwLock<GraphManager>>,
        entity_extraction_enabled: bool,
    ) -> Self {
        Self {
            storage,
            vector_index,
            bm25_index,
            temporal_index,
            graph_manager,
            entity_extractor: SimpleEntityExtractor::new(),
            entity_extraction_enabled,
            #[cfg(feature = "slm")]
            slm_extractor: None,
            #[cfg(feature = "entity-extraction")]
            llm_extractor: None,
            #[cfg(feature = "entity-extraction")]
            llm_extraction_count: std::sync::atomic::AtomicUsize::new(0),
            embedding_fn: None,
            #[cfg(feature = "entity-extraction")]
            extraction_passes: 1,
            profile_entity_types: vec![
                "person".to_string(),
                "organization".to_string(),
                "location".to_string(),
            ],
            async_extraction_threshold: 0,
            #[cfg(feature = "entity-extraction")]
            pending_extractions: Mutex::new(Vec::new()),
        }
    }

    /// Set the SLM metadata extractor for rich metadata extraction at ingestion time
    ///
    /// When set, the pipeline will use the SLM to extract entities, temporal markers,
    /// causal relationships, and topics from memory content during ingestion.
    ///
    /// # Arguments
    ///
    /// * `extractor` - The SLM metadata extractor to use
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mnemefusion_core::ingest::{IngestionPipeline, SlmMetadataExtractor};
    /// use mnemefusion_core::slm::SlmConfig;
    /// use std::sync::{Arc, Mutex};
    ///
    /// // Create pipeline with SLM extractor
    /// let extractor = SlmMetadataExtractor::new(SlmConfig::default()).unwrap();
    /// let pipeline = pipeline.with_slm_extractor(Arc::new(Mutex::new(extractor)));
    /// ```
    #[cfg(feature = "slm")]
    pub fn with_slm_extractor(mut self, extractor: Arc<Mutex<SlmMetadataExtractor>>) -> Self {
        self.slm_extractor = Some(extractor);
        self
    }

    /// Set the native LLM entity extractor for entity extraction at ingestion time
    ///
    /// When set, the pipeline will use the native LLM (via llama.cpp) to extract
    /// entities and facts from memory content during ingestion. This takes
    /// precedence over the Python SLM extractor when both are available.
    ///
    /// # Arguments
    ///
    /// * `extractor` - The LLM entity extractor to use
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mnemefusion_core::extraction::{LlmEntityExtractor, ModelTier};
    /// use std::sync::{Arc, Mutex};
    ///
    /// // Create pipeline with LLM extractor
    /// let extractor = LlmEntityExtractor::load(ModelTier::Balanced).unwrap();
    /// let pipeline = pipeline.with_llm_extractor(Arc::new(Mutex::new(extractor)));
    /// ```
    #[cfg(feature = "entity-extraction")]
    pub fn with_llm_extractor(mut self, extractor: Arc<Mutex<LlmEntityExtractor>>) -> Self {
        self.llm_extractor = Some(extractor);
        self
    }

    /// Run entity extraction on text without adding to the database.
    ///
    /// Returns the extraction result directly for inspection/testing.
    /// Requires `enable_llm_entity_extraction()` to have been called first.
    #[cfg(feature = "entity-extraction")]
    pub fn extract_text(
        &self,
        content: &str,
        speaker: Option<&str>,
    ) -> crate::error::Result<crate::extraction::ExtractionResult> {
        let extractor = self
            .llm_extractor
            .as_ref()
            .ok_or_else(|| {
                crate::error::Error::InferenceError(
                    "LLM extractor not enabled. Call enable_llm_entity_extraction() first."
                        .to_string(),
                )
            })?;
        let ext = extractor.lock().unwrap();
        ext.extract(content, speaker)
    }

    /// Set the embedding function for computing fact embeddings at ingestion time.
    pub fn set_embedding_fn(&mut self, f: EmbeddingFn) {
        self.embedding_fn = Some(f);
    }

    /// Get the current embedding function (if set).
    pub fn embedding_fn(&self) -> Option<EmbeddingFn> {
        self.embedding_fn.clone()
    }

    /// Set the number of LLM extraction passes per document.
    ///
    /// Multiple passes with different temperatures capture different facts,
    /// producing richer entity profiles (similar to ensemble extraction).
    ///
    /// # Arguments
    ///
    /// * `passes` - Number of passes (1 = single pass, 3 = recommended)
    #[cfg(feature = "entity-extraction")]
    pub fn with_extraction_passes(mut self, passes: usize) -> Self {
        self.extraction_passes = passes.max(1);
        self
    }

    /// Set extraction passes on an existing pipeline (mutable reference variant).
    #[cfg(feature = "entity-extraction")]
    pub fn set_extraction_passes(&mut self, passes: usize) {
        self.extraction_passes = passes.max(1);
    }

    /// Set which entity types are allowed to have profiles.
    ///
    /// Entities with types not in this list will be skipped during profile
    /// creation. Empty list = allow all types.
    pub fn set_profile_entity_types(&mut self, types: Vec<String>) {
        self.profile_entity_types = types;
    }

    /// Set the content size threshold for deferred LLM extraction.
    ///
    /// When LLM extraction is enabled and `content.len() >= threshold`, `add()`
    /// stores the memory immediately and queues LLM extraction for later.
    /// Call `flush_extraction_queue()` to process all pending extractions.
    /// Set to `0` (default) to always run LLM extraction synchronously.
    pub fn set_async_extraction_threshold(&mut self, threshold: usize) {
        self.async_extraction_threshold = threshold;
    }

    /// Process all deferred LLM extractions from the pending queue.
    ///
    /// When `async_extraction_threshold > 0`, large `add()` calls skip LLM
    /// extraction and queue work here. This method drains the queue, runs LLM
    /// extraction for each pending memory, and updates entity profiles.
    ///
    /// Returns the number of memories whose extraction was processed.
    ///
    /// Safe to call even if the queue is empty (returns 0).
    /// Thread-safe: queue is locked only briefly for the drain.
    #[cfg(feature = "entity-extraction")]
    pub fn flush_extraction_queue(&self) -> crate::error::Result<usize> {
        let pending = {
            let mut queue = self.pending_extractions.lock().unwrap();
            std::mem::take(&mut *queue)
        };
        let total = pending.len();
        if total == 0 {
            return Ok(0);
        }

        tracing::info!("Flushing {} deferred LLM extractions", total);
        for pending_ex in pending {
            if let Some(ref llm_extractor) = self.llm_extractor {
                let extractor = llm_extractor.lock().unwrap();
                let speaker = pending_ex.speaker.as_deref();
                let session_date = pending_ex.session_date.as_deref();
                let raw_results = if self.extraction_passes == 1 && session_date.is_some() {
                    vec![extractor.extract_typed(&pending_ex.content, speaker, session_date)]
                } else {
                    extractor.extract_multi_pass(
                        &pending_ex.content,
                        speaker,
                        self.extraction_passes,
                    )
                };
                drop(extractor); // release before profile update

                // GPU context auto-reset
                let passes = raw_results.len();
                let prev = self.llm_extraction_count.fetch_add(passes, std::sync::atomic::Ordering::Relaxed);
                if (prev + passes) / Self::GPU_CONTEXT_RESET_INTERVAL > prev / Self::GPU_CONTEXT_RESET_INTERVAL {
                    if let Ok(guard) = llm_extractor.lock() {
                        guard.reset_context();
                    }
                }

                for result in raw_results {
                    match result {
                        Ok(extraction) => {
                            if let Err(e) = self.update_entity_profiles_from_llm(
                                &pending_ex.memory_id,
                                &extraction,
                            ) {
                                tracing::warn!("flush: profile update failed for {}: {}", pending_ex.memory_id, e);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("flush: LLM extraction failed for {}: {}", pending_ex.memory_id, e);
                        }
                    }
                }
            }
        }
        Ok(total)
    }

    /// Returns the number of memories with pending deferred LLM extractions.
    #[cfg(feature = "entity-extraction")]
    pub fn pending_extraction_count(&self) -> usize {
        self.pending_extractions.lock().unwrap().len()
    }

    /// Stub for non-entity-extraction builds.
    #[cfg(not(feature = "entity-extraction"))]
    pub fn flush_extraction_queue(&self) -> crate::error::Result<usize> {
        Ok(0)
    }

    /// Stub for non-entity-extraction builds.
    #[cfg(not(feature = "entity-extraction"))]
    pub fn pending_extraction_count(&self) -> usize {
        0
    }

    /// Reserve capacity in the vector index for future insertions
    ///
    /// This improves performance when adding many memories by avoiding
    /// repeated reallocations.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of vectors to reserve space for
    pub fn reserve_capacity(&self, capacity: usize) -> Result<()> {
        self.vector_index.write().unwrap().reserve(capacity)
    }

    /// Add a memory and index it across all dimensions
    ///
    /// This operation is atomic - if any indexing step fails, all changes
    /// are rolled back and an error is returned.
    ///
    /// # Arguments
    ///
    /// * `memory` - The memory to add (must have valid embedding)
    ///
    /// # Returns
    ///
    /// The MemoryId of the added memory
    ///
    /// # Errors
    ///
    /// Returns error if any indexing step fails. The database state will be
    /// consistent - no partial indexing will occur.
    pub fn add(&self, mut memory: Memory) -> Result<MemoryId> {
        let id = memory.id.clone();
        let timestamp = memory.created_at;

        // Step 0a: Extract temporal expressions from content and store in metadata
        let temporal_extractor = get_temporal_extractor();
        let temporal_expressions = temporal_extractor.extract(&memory.content);
        if !temporal_expressions.is_empty() {
            // Store temporal expressions as JSON array in metadata
            let expressions_json: Vec<String> =
                temporal_expressions.iter().map(|e| e.text().to_string()).collect();
            let json_string = serde_json::to_string(&expressions_json).unwrap_or_default();
            memory.set_metadata("temporal_expressions".to_string(), json_string);
        }

        // Step 0b: Extract causal language patterns from content and store in metadata
        let causal_extractor = get_causal_extractor();
        let (causal_markers, causal_density) = causal_extractor.extract(&memory.content);
        if !causal_markers.is_empty() {
            // Store causal markers as JSON array
            let markers_json = serde_json::to_string(&causal_markers).unwrap_or_default();
            memory.set_metadata("causal_markers".to_string(), markers_json);

            // Store causal density as string
            memory.set_metadata("causal_density".to_string(), causal_density.to_string());
        }

        // Step 0c: Native LLM entity extraction (if available)
        // Uses llama.cpp for entity/fact extraction.
        // Supports multi-pass: runs extraction_passes times with diverse params,
        // merging all extracted facts into profiles via add_fact() dedup.
        // When session_date is available AND single-pass, use typed extraction prompt
        // (episodic/semantic/procedural records + event dates + relationships).
        // This matches the NScale cloud extraction prompt quality.
        //
        // Async mode: when async_extraction_threshold > 0 and content.len() >= threshold,
        // the extraction is deferred. All fast steps (storage, vector, BM25) still run
        // synchronously. Call flush_extraction_queue() to process deferred extractions.
        #[cfg(feature = "entity-extraction")]
        let llm_extraction_results: Vec<ExtractionResult> =
            if let Some(ref llm_extractor) = self.llm_extractor {
                // Check async deferral threshold
                if self.async_extraction_threshold > 0
                    && memory.content.len() >= self.async_extraction_threshold
                {
                    // Defer: queue for flush_extraction_queue()
                    let speaker = memory.metadata.get("speaker").cloned();
                    let session_date = memory.metadata.get("session_date").cloned();
                    self.pending_extractions.lock().unwrap().push(PendingExtraction {
                        memory_id: id.clone(),
                        content: memory.content.clone(),
                        speaker,
                        session_date,
                    });
                    tracing::debug!(
                        "Deferred LLM extraction for memory {} ({} bytes >= threshold {})",
                        id, memory.content.len(), self.async_extraction_threshold
                    );
                    vec![] // No results yet; profile updated later via flush()
                } else {
                    // Sync extraction (existing path)
                let speaker = memory.metadata.get("speaker").map(|s| s.as_str());
                let session_date = memory.metadata.get("session_date").map(|s| s.as_str());
                let extractor = llm_extractor.lock().unwrap();
                let raw_results = if self.extraction_passes == 1 && session_date.is_some() {
                    // Use typed extraction prompt for richer entity facts + temporal metadata.
                    // Research: ENGRAM (arXiv 2511.12960) +31 pts from typed separation,
                    // TReMu (ACL 2025) +47 pts from inferred event dates.
                    vec![extractor.extract_typed(
                        &memory.content,
                        speaker,
                        session_date,
                    )]
                } else {
                    extractor
                        .extract_multi_pass(&memory.content, speaker, self.extraction_passes)
                };
                drop(extractor); // release lock before processing

                let mut successes = Vec::new();
                for (pass, result) in raw_results.into_iter().enumerate() {
                    match result {
                        Ok(extraction) => {
                            tracing::debug!(
                                "LLM pass {}: {} entities, {} topics, {} entity_facts",
                                pass,
                                extraction.entities.len(),
                                extraction.topics.len(),
                                extraction.entity_facts.len()
                            );
                            successes.push(extraction);
                        }
                        Err(e) => {
                            tracing::warn!("LLM extraction pass {} failed: {}", pass, e);
                        }
                    }
                }

                // Store first successful extraction as metadata (backward compat)
                if let Some(first) = successes.first() {
                    let json = serde_json::to_string(first).unwrap_or_default();
                    memory.set_metadata("llm_extraction".to_string(), json);

                    // Collect entity names from ALL passes for backward compat
                    let mut all_names = std::collections::HashSet::new();
                    for extraction in &successes {
                        for entity in &extraction.entities {
                            all_names.insert(entity.name.clone());
                        }
                    }
                    if !all_names.is_empty() {
                        let names: Vec<String> = all_names.into_iter().collect();
                        memory.set_metadata(
                            "entity_names".to_string(),
                            serde_json::to_string(&names).unwrap_or_default(),
                        );
                    }
                }

                    successes
                } // end sync extraction else-branch
            } else {
                Vec::new()
            };

        // Auto-reset GPU context periodically to prevent CUDA memory fragmentation.
        // Count ALL passes (not just documents) for accurate fragmentation tracking.
        #[cfg(feature = "entity-extraction")]
        if !llm_extraction_results.is_empty() {
            let passes_completed = llm_extraction_results.len();
            let count = self.llm_extraction_count.fetch_add(
                passes_completed,
                std::sync::atomic::Ordering::Relaxed,
            ) + passes_completed;
            if count / Self::GPU_CONTEXT_RESET_INTERVAL
                > (count - passes_completed) / Self::GPU_CONTEXT_RESET_INTERVAL
            {
                if let Some(ref extractor) = self.llm_extractor {
                    if let Ok(guard) = extractor.lock() {
                        guard.reset_context();
                        tracing::info!(
                            "Reset GPU context after {} LLM extractions (fragmentation prevention)",
                            count
                        );
                    }
                }
            }
        }

        // Step 0d: Python SLM metadata extraction (fallback if LLM extraction not available)
        // This extracts rich metadata using a Small Language Model for better retrieval
        #[cfg(feature = "slm")]
        let slm_metadata_for_profiles: Option<SlmMetadata> = {
            // Skip if LLM extraction already succeeded
            #[cfg(feature = "entity-extraction")]
            let llm_succeeded = !llm_extraction_results.is_empty();
            #[cfg(not(feature = "entity-extraction"))]
            let llm_succeeded = false;

            if llm_succeeded {
                None // LLM extraction takes precedence
            } else if let Some(ref slm_extractor) = self.slm_extractor {
                match slm_extractor.lock().unwrap().extract(&memory.content) {
                    Ok(slm_metadata) => {
                        // Store full SLM metadata as JSON
                        let json = serde_json::to_string(&slm_metadata).unwrap_or_default();
                        memory.set_metadata("slm_metadata".to_string(), json);

                        // Also populate entity_names for backward compatibility
                        if !slm_metadata.entities.is_empty() {
                            let names: Vec<String> = slm_metadata
                                .entities
                                .iter()
                                .map(|e| e.name.clone())
                                .collect();
                            memory.set_metadata(
                                "entity_names".to_string(),
                                serde_json::to_string(&names).unwrap_or_default(),
                            );
                        }

                        tracing::debug!(
                            "SLM extracted {} entities, {} topics, {} entity_facts from memory",
                            slm_metadata.entities.len(),
                            slm_metadata.topics.len(),
                            slm_metadata.entity_facts.len()
                        );

                        Some(slm_metadata)
                    }
                    Err(e) => {
                        // Log warning but continue - pattern-based extraction already done in 0a/0b
                        tracing::warn!(
                            "SLM extraction failed, using pattern-based extraction: {}",
                            e
                        );
                        None
                    }
                }
            } else {
                None
            }
        };

        // Step 1: Store memory (if this fails, nothing else happens)
        self.storage.store_memory(&memory)?;

        // Step 1b: Update entity profiles from ALL LLM extraction passes
        // Each pass may capture different facts; add_fact() handles dedup.
        #[cfg(feature = "entity-extraction")]
        for extraction in &llm_extraction_results {
            if let Err(e) = self.update_entity_profiles_from_llm(&id, extraction) {
                tracing::warn!("Failed to update entity profiles from LLM: {}", e);
                // Don't fail the entire ingestion - profiles are a bonus feature
            }
        }

        // Step 1c: Update entity profiles from SLM entity_facts (if available and LLM didn't run)
        #[cfg(feature = "slm")]
        if let Some(ref slm_metadata) = slm_metadata_for_profiles {
            if let Err(e) = self.update_entity_profiles_from_slm(&id, slm_metadata) {
                tracing::warn!("Failed to update entity profiles: {}", e);
                // Don't fail the entire ingestion - profiles are a bonus feature
            }
        }

        // Step 2: Add to vector index + persist immediately (rollback: delete from storage)
        if let Err(e) = self.add_to_vector_index(&id, &memory.embedding) {
            // Rollback: remove from storage
            let _ = self.storage.delete_memory(&id);
            return Err(e);
        }

        // Step 2b: Persist vector index immediately for crash recovery
        // This ensures the vector index is always consistent with storage
        if let Err(e) = self.save_vector_index() {
            // Rollback: remove from storage and vector index
            let _ = self.remove_from_vector_index(&id);
            let _ = self.storage.delete_memory(&id);
            return Err(e);
        }

        // Step 2c: Add to BM25 index (rollback: delete from storage + vector)
        if let Err(e) = self.bm25_index.add(&id, &memory.content) {
            // Rollback: remove from storage and vector index
            let _ = self.storage.delete_memory(&id);
            let _ = self.remove_from_vector_index(&id);
            return Err(e);
        }

        // Step 2d: Persist BM25 index for crash recovery
        if let Err(e) = self.bm25_index.save() {
            let _ = self.storage.delete_memory(&id);
            let _ = self.remove_from_vector_index(&id);
            let _ = self.bm25_index.remove(&id);
            return Err(e);
        }

        // Step 3: Add to temporal index (rollback: delete from storage + vector + BM25)
        // Note: Temporal index uses redb storage, so it's already durable
        if let Err(e) = self.temporal_index.add(&id, timestamp) {
            // Rollback: remove from storage, vector index, and BM25 index
            let _ = self.storage.delete_memory(&id);
            let _ = self.remove_from_vector_index(&id);
            let _ = self.bm25_index.remove(&id);
            // Note: Vector index already saved, but we're rolling back the entry
            // The save will be overwritten on next successful operation
            return Err(e);
        }

        // Step 4: Extract and link entities (if enabled)
        if self.entity_extraction_enabled {
            if let Err(e) = self.extract_and_link_entities(&id, &memory.content) {
                // Rollback: remove from all indexes
                let _ = self.storage.delete_memory(&id);
                let _ = self.remove_from_vector_index(&id);
                let _ = self.bm25_index.remove(&id);
                let _ = self.temporal_index.remove(&id);
                return Err(e);
            }

            // Step 4b: Persist graph immediately for crash recovery
            // This ensures the graph is always consistent with storage
            if let Err(e) = self.save_graph() {
                // Rollback: remove from all indexes and clean up entities
                let _ = self.storage.delete_memory(&id);
                let _ = self.remove_from_vector_index(&id);
                let _ = self.bm25_index.remove(&id);
                let _ = self.temporal_index.remove(&id);
                // Note: Entity cleanup is complex, for now we leave stale entities
                // They will be cleaned up on next delete operation
                return Err(e);
            }
        }

        // Step 5: Annotate parent with typed record metadata + store relationships.
        // Note: We do NOT create child memories — they flood the vector index and
        // cause recall collapse (-14.9 pts in S30 testing). Instead, typed decomposition
        // is stored as metadata on the parent for type-aware retrieval balancing.
        #[cfg(feature = "entity-extraction")]
        for extraction in &llm_extraction_results {
            if !extraction.records.is_empty() {
                self.annotate_parent_with_types(&id, &extraction.records);
            }
            if !extraction.relationships.is_empty() {
                if let Err(e) = self.store_relationships(&id, &extraction.relationships) {
                    tracing::warn!("Failed to store relationships: {}", e);
                }
            }
        }

        Ok(id)
    }

    /// Delete a memory and clean up all indexes
    ///
    /// This removes the memory from:
    /// - Storage
    /// - Vector index
    /// - Temporal index
    /// - Entity graph
    /// - Causal graph
    ///
    /// Orphaned entities (entities with no remaining mentions) are also removed.
    ///
    /// # Arguments
    ///
    /// * `id` - The MemoryId to delete
    ///
    /// # Returns
    ///
    /// true if the memory was deleted, false if it didn't exist
    ///
    /// # Errors
    ///
    /// Returns error if any deletion step fails. The database may be in an
    /// inconsistent state if an error occurs during cleanup.
    pub fn delete(&self, id: &MemoryId) -> Result<bool> {
        // Check if memory exists
        let _memory = match self.storage.get_memory(id)? {
            Some(mem) => mem,
            None => return Ok(false), // Memory doesn't exist
        };

        // Get entities linked to this memory BEFORE deleting
        let entity_ids = {
            let graph = self.graph_manager.read().unwrap();
            graph.get_memory_entities(id)
        };

        // Step 1: Delete from storage
        self.storage.delete_memory(id)?;

        // Step 2: Remove from vector index
        // Ignore errors - index might not have the entry
        let _ = self.remove_from_vector_index(id);

        // Step 3: Remove from BM25 index
        // Ignore errors - index might not have the entry
        let _ = self.bm25_index.remove(id);

        // Step 4: Remove from temporal index
        // Ignore errors - index might not have the entry
        let _ = self.temporal_index.remove(id);

        // Step 5: Remove from entity graph and clean up orphaned entities
        {
            let mut graph = self.graph_manager.write().unwrap();
            graph.remove_memory_from_entity_graph(id);
        }

        // Check each entity to see if it's now orphaned
        for entity_id in entity_ids {
            if let Some(mut entity) = self.storage.get_entity(&entity_id)? {
                // Decrement mention count
                entity.decrement_mentions();

                if entity.mention_count == 0 {
                    // Entity is orphaned, remove it
                    self.storage.delete_entity(&entity_id)?;

                    // Remove from entity graph
                    let mut graph = self.graph_manager.write().unwrap();
                    graph.remove_entity_from_graph(&entity_id);
                } else {
                    // Entity still has mentions, just update the count
                    self.storage.store_entity(&entity)?;
                }
            }
        }

        // Step 5b: Remove from causal graph
        {
            let mut graph = self.graph_manager.write().unwrap();
            graph.remove_memory_from_causal_graph(id);
        }

        // Step 5c: Clean up entity profiles
        // Remove any facts that came from this memory
        self.cleanup_entity_profiles_for_memory(id)?;

        // Step 6: Persist vector index for crash recovery
        // This ensures deletions are durable
        self.save_vector_index()?;

        // Step 7: Persist graph for crash recovery
        // This ensures entity and causal graph updates are durable
        self.save_graph()?;

        Ok(true)
    }

    /// Add multiple memories in a batch operation
    ///
    /// This is significantly faster than calling `add()` multiple times because:
    /// - Single transaction for all storage operations
    /// - Vector index is locked once for all additions
    /// - Entity extraction is deduplicated across the entire batch
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of MemoryInput to add
    /// * `progress_callback` - Optional callback for progress updates: fn(current, total)
    ///
    /// # Returns
    ///
    /// BatchResult containing IDs of created memories and any errors
    ///
    /// # Performance
    ///
    /// Target: 1,000 memories in <500ms (10x+ faster than individual adds)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let inputs = vec![
    ///     MemoryInput::new("content 1".into(), vec![0.1; 384]),
    ///     MemoryInput::new("content 2".into(), vec![0.2; 384]),
    /// ];
    ///
    /// let result = pipeline.add_batch(inputs, None)?;
    /// println!("Created {} memories", result.created_count);
    /// ```
    pub fn add_batch(
        &self,
        inputs: Vec<MemoryInput>,
        progress_callback: Option<Box<dyn Fn(usize, usize)>>,
    ) -> Result<BatchResult> {
        let mut result = BatchResult::new();
        let total = inputs.len();

        // Convert inputs to memories and extract temporal/causal expressions
        let temporal_extractor = get_temporal_extractor();
        let causal_extractor = get_causal_extractor();

        let mut memories: Vec<Memory> = inputs
            .iter()
            .map(|input| {
                let mut memory = input.to_memory();

                // Extract temporal expressions from content
                let temporal_expressions = temporal_extractor.extract(&memory.content);
                if !temporal_expressions.is_empty() {
                    let expressions_json: Vec<String> =
                        temporal_expressions.iter().map(|e| e.text().to_string()).collect();
                    let json_string = serde_json::to_string(&expressions_json).unwrap_or_default();
                    memory.set_metadata("temporal_expressions".to_string(), json_string);
                }

                // Extract causal language patterns from content
                let (causal_markers, causal_density) = causal_extractor.extract(&memory.content);
                if !causal_markers.is_empty() {
                    let markers_json = serde_json::to_string(&causal_markers).unwrap_or_default();
                    memory.set_metadata("causal_markers".to_string(), markers_json);
                    memory.set_metadata("causal_density".to_string(), causal_density.to_string());
                }

                memory
            })
            .collect();

        // Note: LLM/SLM extraction is NOT done in batch mode by design.
        // Per architecture decision, model inference should be handled at the application layer.
        // Use individual add() calls if you need LLM extraction.
        // Batch add is optimized for fast bulk ingestion without extraction.

        // Lock vector index once for the entire batch
        let mut vector_index = self.vector_index.write().unwrap();

        // Process each memory
        for (index, memory) in memories.iter().enumerate() {
            let id = memory.id.clone();
            let timestamp = memory.created_at;

            // Report progress
            if let Some(ref callback) = progress_callback {
                callback(index + 1, total);
            }

            // Step 1: Store memory
            if let Err(e) = self.storage.store_memory(memory) {
                result
                    .errors
                    .push(BatchError::new(index, format!("Storage failed: {}", e)));
                continue;
            }

            // Step 2: Add to vector index (index already locked)
            if let Err(e) = vector_index.add(id.clone(), &memory.embedding) {
                result.errors.push(BatchError::with_id(
                    index,
                    format!("Vector index failed: {}", e),
                    id.clone(),
                ));
                // Rollback: remove from storage
                let _ = self.storage.delete_memory(&id);
                continue;
            }

            // Step 3: Add to temporal index
            if let Err(e) = self.temporal_index.add(&id, timestamp) {
                result.errors.push(BatchError::with_id(
                    index,
                    format!("Temporal index failed: {}", e),
                    id.clone(),
                ));
                // Rollback: remove from storage and vector index
                let _ = self.storage.delete_memory(&id);
                let _ = vector_index.remove(&id);
                continue;
            }

            // Step 4: Add to BM25 index
            if let Err(e) = self.bm25_index.add(&id, &memory.content) {
                result.errors.push(BatchError::with_id(
                    index,
                    format!("BM25 index failed: {}", e),
                    id.clone(),
                ));
                // Rollback: remove from storage, vector, and temporal indexes
                let _ = self.storage.delete_memory(&id);
                let _ = vector_index.remove(&id);
                let _ = self.temporal_index.remove(&id);
                continue;
            }

            // Step 5: Extract and link entities (if enabled)
            if self.entity_extraction_enabled {
                if let Err(e) = self.extract_and_link_entities(&id, &memory.content) {
                    result.errors.push(BatchError::with_id(
                        index,
                        format!("Entity extraction failed: {}", e),
                        id.clone(),
                    ));
                    // Rollback: remove from all indexes
                    let _ = self.storage.delete_memory(&id);
                    let _ = vector_index.remove(&id);
                    let _ = self.temporal_index.remove(&id);
                    let _ = self.bm25_index.remove(&id);
                    continue;
                }
            }

            // Success! Add to result
            result.ids.push(id);
            result.created_count += 1;
        }

        // Release vector index lock
        drop(vector_index);

        // Persist vector index once for entire batch
        // This is much more efficient than saving per-memory
        self.save_vector_index()?;

        // Persist BM25 index once for entire batch
        if result.created_count > 0 {
            self.bm25_index.save()?;
        }

        // Persist graph once for entire batch (if entity extraction was enabled)
        if self.entity_extraction_enabled && result.created_count > 0 {
            self.save_graph()?;
        }

        Ok(result)
    }

    /// Delete multiple memories in a batch operation
    ///
    /// This is faster than calling `delete()` multiple times because:
    /// - Single transaction for all storage operations
    /// - Entity cleanup is batched
    ///
    /// # Arguments
    ///
    /// * `ids` - Vector of MemoryIds to delete
    ///
    /// # Returns
    ///
    /// Number of memories actually deleted (may be less than input if some don't exist)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let ids = vec![id1, id2, id3];
    /// let deleted_count = pipeline.delete_batch(ids)?;
    /// println!("Deleted {} memories", deleted_count);
    /// ```
    pub fn delete_batch(&self, ids: Vec<MemoryId>) -> Result<usize> {
        let mut deleted_count = 0;

        // Collect all entity IDs from memories before deletion
        let mut all_entity_ids = std::collections::HashSet::new();
        for id in &ids {
            let entity_ids = {
                let graph = self.graph_manager.read().unwrap();
                graph.get_memory_entities(id)
            };
            all_entity_ids.extend(entity_ids);
        }

        // Lock vector index once for the entire batch
        let mut vector_index = self.vector_index.write().unwrap();

        // Delete each memory
        for id in &ids {
            // Check if memory exists
            if self.storage.get_memory(id)?.is_none() {
                continue; // Memory doesn't exist, skip
            }

            // Step 1: Delete from storage
            if let Err(_e) = self.storage.delete_memory(id) {
                // If storage deletion fails, skip this memory
                continue;
            }

            // Step 2: Remove from vector index (already locked)
            let _ = vector_index.remove(id);

            // Step 3: Remove from temporal index
            let _ = self.temporal_index.remove(id);

            // Step 4: Remove from entity graph
            {
                let mut graph = self.graph_manager.write().unwrap();
                graph.remove_memory_from_entity_graph(id);
            }

            // Step 5: Remove from causal graph
            {
                let mut graph = self.graph_manager.write().unwrap();
                graph.remove_memory_from_causal_graph(id);
            }

            deleted_count += 1;
        }

        // Release vector index lock
        drop(vector_index);

        // Clean up orphaned entities (batch operation)
        for entity_id in all_entity_ids {
            if let Some(mut entity) = self.storage.get_entity(&entity_id)? {
                // Get current mention count from graph
                let graph = self.graph_manager.read().unwrap();
                let result = graph.get_entity_memories(&entity_id);
                drop(graph);

                // Update entity mention count
                entity.mention_count = result.memories.len();

                if entity.mention_count == 0 {
                    // Entity is orphaned, remove it
                    self.storage.delete_entity(&entity_id)?;

                    // Remove from entity graph
                    let mut graph = self.graph_manager.write().unwrap();
                    graph.remove_entity_from_graph(&entity_id);
                } else {
                    // Entity still has mentions, update the count
                    self.storage.store_entity(&entity)?;
                }
            }
        }

        // Clean up entity profiles for all deleted memories
        for id in &ids {
            self.cleanup_entity_profiles_for_memory(id)?;
        }

        // Persist vector index once for entire batch
        // This is much more efficient than saving per-memory
        if deleted_count > 0 {
            self.save_vector_index()?;

            // Persist graph once for entire batch
            self.save_graph()?;
        }

        Ok(deleted_count)
    }

    /// Add a memory with deduplication
    ///
    /// Checks if identical content already exists using content hash.
    /// If duplicate found, returns existing ID without storing.
    /// If unique, stores normally.
    ///
    /// # Arguments
    ///
    /// * `memory` - The memory to add
    ///
    /// # Returns
    ///
    /// AddResult indicating whether memory was created or duplicate found
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let memory = Memory::new("test content".into(), vec![0.1; 384]);
    /// let result = pipeline.add_with_dedup(memory)?;
    ///
    /// if result.created {
    ///     println!("New memory: {}", result.id);
    /// } else {
    ///     println!("Duplicate found: {}", result.existing_id.unwrap());
    /// }
    /// ```
    pub fn add_with_dedup(&self, memory: Memory) -> Result<AddResult> {
        let content_hash = hash::hash_content(&memory.content);

        // Check if content hash already exists
        if let Some(existing_id) = self.storage.find_by_content_hash(&content_hash)? {
            // Verify it's actually the same content (handle hash collisions)
            if let Some(existing_memory) = self.storage.get_memory(&existing_id)? {
                if existing_memory.content == memory.content {
                    // True duplicate found
                    return Ok(AddResult::duplicate(existing_id));
                }
                // Hash collision - different content, same hash (extremely rare)
                // Fall through to add as new memory
            }
        }

        // No duplicate, add normally
        let id = self.add(memory)?;

        // Store content hash mapping
        self.storage.store_content_hash(&content_hash, &id)?;

        Ok(AddResult::created(id))
    }

    /// Upsert a memory by logical key
    ///
    /// If key exists: replaces content, embedding, and metadata of existing memory
    /// If key doesn't exist: creates new memory and associates with key
    ///
    /// This is an atomic operation - either everything succeeds or everything rolls back.
    ///
    /// # Arguments
    ///
    /// * `key` - Logical key for the memory (e.g., "user_profile:123")
    /// * `memory` - The memory data to upsert
    ///
    /// # Returns
    ///
    /// UpsertResult indicating whether memory was created or updated
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let memory = Memory::new("updated content".into(), vec![0.2; 384]);
    /// let result = pipeline.upsert("doc:readme", memory)?;
    ///
    /// if result.created {
    ///     println!("Created new memory");
    /// } else {
    ///     println!("Updated existing memory");
    /// }
    /// ```
    pub fn upsert(&self, key: &str, memory: Memory) -> Result<UpsertResult> {
        // Check if logical key already exists
        if let Some(existing_id) = self.storage.find_by_logical_key(key)? {
            // Key exists - this is an update
            let previous_memory = self.storage.get_memory(&existing_id)?;
            let previous_content = previous_memory.as_ref().map(|m| m.content.clone());

            // Delete the old memory completely (including all indexes)
            self.delete(&existing_id)?;

            // Add the new memory with the same logical key
            let new_id = self.add(memory)?;

            // Update the logical key mapping to point to new memory
            self.storage.update_logical_key(key, &new_id)?;

            Ok(UpsertResult::updated(new_id, previous_content))
        } else {
            // Key doesn't exist - this is a create
            let id = self.add(memory)?;

            // Store the logical key mapping
            self.storage.store_logical_key(key, &id)?;

            Ok(UpsertResult::created(id))
        }
    }

    /// Add memory to vector index
    fn add_to_vector_index(&self, id: &MemoryId, embedding: &[f32]) -> Result<()> {
        let mut index = self.vector_index.write().unwrap();
        index.add(id.clone(), embedding)
    }

    /// Remove memory from vector index
    fn remove_from_vector_index(&self, id: &MemoryId) -> Result<()> {
        let mut index = self.vector_index.write().unwrap();
        index.remove(id)
    }

    /// Extract entities from content and link to memory
    fn extract_and_link_entities(&self, memory_id: &MemoryId, content: &str) -> Result<()> {
        // Extract entity names
        let entity_names = self.entity_extractor.extract(content)?;

        // Deduplicate entity names (case-insensitive)
        let mut seen = std::collections::HashSet::new();
        let unique_entities: Vec<&String> = entity_names
            .iter()
            .filter(|name| seen.insert(name.to_lowercase()))
            .collect();

        // Store entity names in memory metadata for content-based matching
        if !unique_entities.is_empty() {
            let entity_names_json: Vec<String> =
                unique_entities.iter().map(|s| s.to_string()).collect();
            let json_string = serde_json::to_string(&entity_names_json).unwrap_or_default();

            // Update memory metadata with entity names
            if let Some(mut memory) = self.storage.get_memory(memory_id)? {
                memory.set_metadata("entity_names".to_string(), json_string);
                self.storage.store_memory(&memory)?;
            }
        }

        // Process each unique extracted entity
        for entity_name in unique_entities {
            // Check if entity already exists
            let entity = match self.storage.find_entity_by_name(entity_name)? {
                Some(mut existing_entity) => {
                    // Entity exists, increment mention count
                    existing_entity.increment_mentions();
                    self.storage.store_entity(&existing_entity)?;
                    existing_entity
                }
                None => {
                    // Create new entity (starts with mention_count = 0)
                    let mut new_entity = Entity::new(entity_name.clone());
                    new_entity.increment_mentions();
                    self.storage.store_entity(&new_entity)?;
                    new_entity
                }
            };

            // Link memory to entity in graph
            let mut graph = self.graph_manager.write().unwrap();
            graph.link_memory_to_entity(memory_id, &entity.id);
        }

        Ok(())
    }

    /// Save vector index to storage
    ///
    /// This ensures the vector index is durable and consistent with storage state.
    /// Called immediately after vector index modifications to prevent data loss on crash.
    fn save_vector_index(&self) -> Result<()> {
        let index = self.vector_index.read().unwrap();
        index.save()
    }

    /// Save graph state to storage
    ///
    /// This ensures the graph is durable and consistent with storage state.
    /// Called immediately after graph modifications to prevent data loss on crash.
    fn save_graph(&self) -> Result<()> {
        let graph = self.graph_manager.read().unwrap();
        crate::graph::persist::save_graph(&graph, &self.storage)
    }

    /// Update entity profiles from LLM-extracted entity facts
    ///
    /// This creates or updates EntityProfiles based on the entity_facts extracted
    /// by the native LLM at ingestion time. Each fact is associated with its source
    /// memory for provenance tracking.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory that the facts were extracted from
    /// * `extraction` - The LLM extraction result containing entity_facts
    /// Check if an entity type is allowed to have a profile.
    ///
    /// If `profile_entity_types` is empty, all types are allowed.
    /// Otherwise, the entity_type must match (case-insensitive) one of the allowed types.
    fn is_entity_type_allowed(&self, entity_type: &str) -> bool {
        if self.profile_entity_types.is_empty() {
            return true;
        }
        let et_lower = entity_type.to_lowercase();
        self.profile_entity_types.iter().any(|t| t.to_lowercase() == et_lower)
    }

    /// Update entity profiles from an extraction result.
    ///
    /// This is the core profile update logic: resolves aliases, filters entity types,
    /// creates/updates profiles, computes fact embeddings, and links source memories.
    /// Used internally during ingestion and externally via `MemoryEngine::apply_extraction()`
    /// for API-based extraction backends (e.g., NScale cloud inference).
    #[cfg(feature = "entity-extraction")]
    pub fn update_entity_profiles_from_llm(
        &self,
        memory_id: &MemoryId,
        extraction: &ExtractionResult,
    ) -> Result<()> {
        // Load known profile names for alias resolution (mutable: new profiles are pushed
        // so later entities in the same extraction can resolve aliases to them)
        let mut known_names = self.storage.list_entity_profile_names()?;

        // Track which entities we've already linked via entity_facts
        let mut linked_entities = std::collections::HashSet::new();

        for extracted_fact in &extraction.entity_facts {
            // Determine entity type from extraction entities if available
            let entity_type = extraction
                .entities
                .iter()
                .find(|e| e.name.eq_ignore_ascii_case(&extracted_fact.entity))
                .map(|e| e.entity_type.clone())
                .unwrap_or_else(|| "unknown".to_string());

            // Skip entities whose type is not in the allowed list
            if !self.is_entity_type_allowed(&entity_type) {
                // Still check if there's an existing profile (might be from before filter was added)
                let entity_name = crate::query::profile_search::resolve_entity_alias(
                    &extracted_fact.entity,
                    &known_names,
                )
                .unwrap_or_else(|| extracted_fact.entity.clone());
                if self.storage.get_entity_profile(&entity_name)?.is_none() {
                    tracing::debug!(
                        "Skipping entity '{}' (type '{}') — not in allowed profile types",
                        extracted_fact.entity,
                        entity_type,
                    );
                    continue;
                }
            }

            // Canonicalize entity name via alias resolution
            let entity_name = crate::query::profile_search::resolve_entity_alias(
                &extracted_fact.entity,
                &known_names,
            )
            .unwrap_or_else(|| extracted_fact.entity.clone());

            // Get or create profile for this entity
            let mut profile = self
                .storage
                .get_entity_profile(&entity_name)?
                .unwrap_or_else(|| {
                    EntityProfile::new(
                        EntityId::new(),
                        entity_name.clone(),
                        entity_type.clone(),
                    )
                });

            // Create EntityFact from ExtractedFact
            let fact = EntityFact {
                fact_type: extracted_fact.fact_type.clone(),
                value: extracted_fact.value.clone(),
                confidence: extracted_fact.confidence,
                source_memory: memory_id.clone(),
                extracted_at: Timestamp::now(),
            };

            // Add fact to profile
            profile.add_fact(fact);

            // Compute and store fact embedding if embedding_fn is available
            if let Some(ref embed_fn) = self.embedding_fn {
                let fact_text = format!(
                    "{} {}",
                    extracted_fact.fact_type.replace('_', " "),
                    extracted_fact.value
                );
                let embedding = embed_fn(&fact_text);
                let key = fact_embedding_key(
                    &entity_name,
                    &extracted_fact.fact_type,
                    &extracted_fact.value,
                );
                if let Err(e) = self.storage.store_fact_embedding(&key, &embedding) {
                    tracing::warn!("Failed to store fact embedding: {}", e);
                }
            }

            // Track source memory
            profile.add_source_memory(memory_id.clone());

            // Save profile
            self.storage.store_entity_profile(&profile)?;

            // Register newly-created profile so later entities can resolve aliases to it
            let entity_name_lower = entity_name.to_lowercase();
            if !known_names.contains(&entity_name_lower) {
                known_names.push(entity_name_lower.clone());
            }

            linked_entities.insert(entity_name_lower);

            tracing::debug!(
                "Updated entity profile '{}' with LLM fact: {} = {} (raw: '{}')",
                entity_name,
                extracted_fact.fact_type,
                extracted_fact.value,
                extracted_fact.entity,
            );
        }

        // Link source_memory for entities that were detected but had no entity_facts.
        // Without this, memories mentioning an entity (e.g. "Caroline") but producing
        // no structured facts would never appear in that entity's source_memories,
        // making them invisible to Step 2.1's entity scoring (the sacred 2.0 baseline).
        for entity in &extraction.entities {
            // Skip entities whose type is not in the allowed list
            // (unless they already have a profile from before the filter was added)
            if !self.is_entity_type_allowed(&entity.entity_type) {
                let resolved = crate::query::profile_search::resolve_entity_alias(
                    &entity.name,
                    &known_names,
                )
                .unwrap_or_else(|| entity.name.clone());
                if self.storage.get_entity_profile(&resolved)?.is_none() {
                    continue;
                }
            }

            // Canonicalize entity name via alias resolution
            let entity_name = crate::query::profile_search::resolve_entity_alias(
                &entity.name,
                &known_names,
            )
            .unwrap_or_else(|| entity.name.clone());

            if linked_entities.contains(&entity_name.to_lowercase()) {
                continue; // Already linked via entity_facts above
            }

            let mut profile = self
                .storage
                .get_entity_profile(&entity_name)?
                .unwrap_or_else(|| {
                    EntityProfile::new(
                        EntityId::new(),
                        entity_name.clone(),
                        entity.entity_type.clone(),
                    )
                });

            profile.add_source_memory(memory_id.clone());
            self.storage.store_entity_profile(&profile)?;

            // Register newly-created profile so later entities can resolve aliases to it
            let entity_name_lower = entity_name.to_lowercase();
            if !known_names.contains(&entity_name_lower) {
                known_names.push(entity_name_lower);
            }

            tracing::debug!(
                "Linked source memory to entity profile '{}' (no facts, entity-only, raw: '{}')",
                entity_name,
                entity.name,
            );
        }

        Ok(())
    }

    /// Create child memories from typed records extracted by the ENGRAM-inspired prompt.
    ///
    /// Each `TypedRecord` in the extraction result becomes a separate Memory with:
    /// - content = record.summary (self-contained sentence)
    /// - metadata["record_type"] = "episodic" | "semantic" | "procedural"
    /// - metadata["record_type"] = primary type ("episodic", "semantic", "procedural")
    /// - metadata["event_date"] = earliest ISO-8601 date from episodic records
    /// - metadata["typed_summaries"] = JSON array of self-contained summaries
    ///
    /// Research basis: ENGRAM (arXiv 2511.12960) shows +31 pts from typed separation.
    /// However, creating child memories in the same vector index causes recall collapse
    /// (-14.9 pts in S30 testing). Instead, we annotate parents with type metadata
    /// for type-aware retrieval balancing (Phase 3) without inflating the index.
    #[cfg(feature = "entity-extraction")]
    pub fn annotate_parent_with_types(
        &self,
        parent_id: &MemoryId,
        records: &[crate::extraction::TypedRecord],
    ) {
        if records.is_empty() {
            return;
        }

        let mut parent = match self.storage.get_memory(parent_id) {
            Ok(Some(p)) => p,
            _ => return,
        };

        // Determine primary type by priority: episodic > procedural > semantic
        // (episodic has event_date which helps temporal, procedural is rare and distinctive)
        let mut has_episodic = false;
        let mut has_procedural = false;
        let mut earliest_event_date: Option<String> = None;
        let mut summaries: Vec<String> = Vec::new();

        for record in records {
            if record.summary.trim().is_empty() {
                continue;
            }
            match record.record_type.as_str() {
                "episodic" => {
                    has_episodic = true;
                    if let Some(ref date) = record.event_date {
                        match &earliest_event_date {
                            None => earliest_event_date = Some(date.clone()),
                            Some(existing) if date < existing => {
                                earliest_event_date = Some(date.clone())
                            }
                            _ => {}
                        }
                    }
                }
                "procedural" => has_procedural = true,
                _ => {} // semantic is the default
            }
            summaries.push(record.summary.clone());
        }

        let primary_type = if has_episodic {
            "episodic"
        } else if has_procedural {
            "procedural"
        } else {
            "semantic"
        };

        parent.set_metadata("record_type".to_string(), primary_type.to_string());

        if let Some(ref date) = earliest_event_date {
            parent.set_metadata("event_date".to_string(), date.clone());
        }

        if !summaries.is_empty() {
            if let Ok(json) = serde_json::to_string(&summaries) {
                parent.set_metadata("typed_summaries".to_string(), json);
            }
        }

        if let Err(e) = self.storage.store_memory(&parent) {
            tracing::warn!("Failed to annotate parent {} with types: {}", parent_id, e);
        }

        tracing::debug!(
            "Annotated parent {} as {} ({} summaries, event_date={:?})",
            parent_id,
            primary_type,
            summaries.len(),
            earliest_event_date
        );
    }

    /// Store extracted entity-to-entity relationships in entity profiles.
    ///
    /// Each relationship is stored as a "relationship" fact in both entities' profiles.
    /// For example, Alice→Bob "spouse" creates:
    /// - Alice profile: fact_type="relationship", value="spouse of Bob"
    /// - Bob profile: fact_type="relationship", value="spouse of Alice"
    #[cfg(feature = "entity-extraction")]
    pub fn store_relationships(
        &self,
        memory_id: &MemoryId,
        relationships: &[crate::extraction::ExtractedRelationship],
    ) -> Result<()> {
        let known_names = self.storage.list_entity_profile_names()?;

        for rel in relationships {
            if rel.confidence < 0.5 {
                continue; // Skip low-confidence relationships
            }

            // Resolve aliases for both entities
            let from_name = crate::query::profile_search::resolve_entity_alias(
                &rel.from_entity,
                &known_names,
            )
            .unwrap_or_else(|| rel.from_entity.clone());

            let to_name = crate::query::profile_search::resolve_entity_alias(
                &rel.to_entity,
                &known_names,
            )
            .unwrap_or_else(|| rel.to_entity.clone());

            // Store relationship fact in "from" entity's profile
            if let Ok(Some(mut from_profile)) = self.storage.get_entity_profile(&from_name) {
                let fact = EntityFact {
                    fact_type: "relationship".to_string(),
                    value: format!("{} of {}", rel.relation_type, to_name),
                    confidence: rel.confidence,
                    source_memory: memory_id.clone(),
                    extracted_at: Timestamp::now(),
                };
                from_profile.add_fact(fact);
                from_profile.add_source_memory(memory_id.clone());
                self.storage.store_entity_profile(&from_profile)?;
            }

            // Store reciprocal relationship fact in "to" entity's profile
            if let Ok(Some(mut to_profile)) = self.storage.get_entity_profile(&to_name) {
                let fact = EntityFact {
                    fact_type: "relationship".to_string(),
                    value: format!("{} of {}", rel.relation_type, from_name),
                    confidence: rel.confidence,
                    source_memory: memory_id.clone(),
                    extracted_at: Timestamp::now(),
                };
                to_profile.add_fact(fact);
                to_profile.add_source_memory(memory_id.clone());
                self.storage.store_entity_profile(&to_profile)?;
            }

            // Add entity-to-entity edge in the graph for 1-hop traversal
            let from_profile = self.storage.get_entity_profile(&from_name)?;
            let to_profile = self.storage.get_entity_profile(&to_name)?;
            if let (Some(fp), Some(tp)) = (from_profile, to_profile) {
                let mut graph = self.graph_manager.write().unwrap();
                graph.link_entity_to_entity(&fp.entity_id, &tp.entity_id, &rel.relation_type);
            }

            tracing::debug!(
                "Stored relationship: {} --[{}]--> {} (confidence: {:.2})",
                from_name,
                rel.relation_type,
                to_name,
                rel.confidence,
            );
        }

        Ok(())
    }

    /// Update entity profiles from SLM-extracted entity facts
    ///
    /// This creates or updates EntityProfiles based on the entity_facts extracted
    /// by the SLM at ingestion time. Each fact is associated with its source memory
    /// for provenance tracking.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory that the facts were extracted from
    /// * `slm_metadata` - The SLM metadata containing entity_facts
    #[cfg(feature = "slm")]
    fn update_entity_profiles_from_slm(
        &self,
        memory_id: &MemoryId,
        slm_metadata: &SlmMetadata,
    ) -> Result<()> {
        if slm_metadata.entity_facts.is_empty() {
            return Ok(());
        }

        for extracted_fact in &slm_metadata.entity_facts {
            // Get or create profile for this entity
            let mut profile = self
                .storage
                .get_entity_profile(&extracted_fact.entity)?
                .unwrap_or_else(|| {
                    // Determine entity type from SLM entities if available
                    let entity_type = slm_metadata
                        .entities
                        .iter()
                        .find(|e| e.name.eq_ignore_ascii_case(&extracted_fact.entity))
                        .map(|e| e.entity_type.clone())
                        .unwrap_or_else(|| "unknown".to_string());

                    EntityProfile::new(
                        EntityId::new(),
                        extracted_fact.entity.clone(),
                        entity_type,
                    )
                });

            // Create EntityFact from ExtractedEntityFact
            let fact = EntityFact {
                fact_type: extracted_fact.fact_type.clone(),
                value: extracted_fact.value.clone(),
                confidence: extracted_fact.confidence,
                source_memory: memory_id.clone(),
                extracted_at: Timestamp::now(),
            };

            // Add fact to profile
            profile.add_fact(fact);

            // Compute and store fact embedding if embedding_fn is available
            if let Some(ref embed_fn) = self.embedding_fn {
                let fact_text = format!(
                    "{} {}",
                    extracted_fact.fact_type.replace('_', " "),
                    extracted_fact.value
                );
                let embedding = embed_fn(&fact_text);
                let key = fact_embedding_key(
                    &extracted_fact.entity,
                    &extracted_fact.fact_type,
                    &extracted_fact.value,
                );
                if let Err(e) = self.storage.store_fact_embedding(&key, &embedding) {
                    tracing::warn!("Failed to store fact embedding: {}", e);
                }
            }

            // Track source memory
            profile.add_source_memory(memory_id.clone());

            // Save profile
            self.storage.store_entity_profile(&profile)?;

            tracing::debug!(
                "Updated entity profile '{}' with fact: {} = {}",
                extracted_fact.entity,
                extracted_fact.fact_type,
                extracted_fact.value
            );
        }

        Ok(())
    }

    /// Clean up entity profiles when a memory is deleted
    ///
    /// This removes any facts from entity profiles that were extracted from
    /// the deleted memory. If a profile becomes empty after cleanup, it is deleted.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory that was deleted
    fn cleanup_entity_profiles_for_memory(&self, memory_id: &MemoryId) -> Result<()> {
        // Get all entity profiles
        let profiles = self.storage.list_entity_profiles()?;

        for mut profile in profiles {
            // Check if this profile has facts from the deleted memory
            if !profile.source_memories.contains(memory_id) {
                continue;
            }

            // Remove facts from this memory
            profile.remove_facts_from_memory(memory_id);

            // Check if profile is now empty
            if profile.is_empty() {
                // Delete the empty profile
                self.storage.delete_entity_profile(&profile.name)?;
                tracing::debug!(
                    "Deleted empty entity profile '{}' after memory deletion",
                    profile.name
                );
            } else {
                // Save the updated profile
                self.storage.store_entity_profile(&profile)?;
                tracing::debug!(
                    "Removed facts from profile '{}' for deleted memory",
                    profile.name
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::GraphManager,
        index::{TemporalIndex, VectorIndex, VectorIndexConfig},
        storage::StorageEngine,
        types::{Memory, Timestamp},
    };
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_test_pipeline() -> (IngestionPipeline, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let storage = Arc::new(StorageEngine::open(&path).unwrap());

        let vector_config = VectorIndexConfig {
            dimension: 384,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        };
        let vector_index = Arc::new(RwLock::new(
            VectorIndex::new(vector_config, Arc::clone(&storage)).unwrap(),
        ));

        let bm25_index = Arc::new(crate::index::BM25Index::new(
            Arc::clone(&storage),
            crate::index::BM25Config::default(),
        ));

        let temporal_index = Arc::new(TemporalIndex::new(Arc::clone(&storage)));
        let graph_manager = Arc::new(RwLock::new(GraphManager::new()));

        let pipeline = IngestionPipeline::new(
            storage,
            vector_index,
            bm25_index,
            temporal_index,
            graph_manager,
            true, // Enable entity extraction
        );

        (pipeline, dir)
    }

    #[test]
    fn test_pipeline_add_and_retrieve() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();

        // Add memory
        let returned_id = pipeline.add(memory).unwrap();
        assert_eq!(id, returned_id);

        // Verify it's in storage
        let retrieved = pipeline.storage.get_memory(&id).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_pipeline_add_indexes_all_dimensions() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();
        let timestamp = memory.created_at;

        pipeline.add(memory).unwrap();

        // Verify vector index
        {
            let index = pipeline.vector_index.read().unwrap();
            let results = index.search(&vec![0.1; 384], 10).unwrap();
            assert!(!results.is_empty());
        }

        // Verify temporal index (use wider time range)
        let start = Timestamp::from_unix_secs(timestamp.as_unix_secs() - 1.0);
        let end = Timestamp::from_unix_secs(timestamp.as_unix_secs() + 1.0);
        let temporal_results = pipeline.temporal_index.range_query(start, end, 10).unwrap();
        assert_eq!(temporal_results.len(), 1);
        assert_eq!(temporal_results[0].id, id);
    }

    #[test]
    fn test_pipeline_add_extracts_entities() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("Alice met Bob in Paris".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();

        pipeline.add(memory).unwrap();

        // Check that entities were created
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());

        let bob = pipeline.storage.find_entity_by_name("Bob").unwrap();
        assert!(bob.is_some());

        let paris = pipeline.storage.find_entity_by_name("Paris").unwrap();
        assert!(paris.is_some());

        // Check that entities are linked in graph
        let graph = pipeline.graph_manager.read().unwrap();
        let entities = graph.get_memory_entities(&id);
        assert_eq!(entities.len(), 3);
    }

    #[test]
    fn test_pipeline_delete_removes_from_all_indexes() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();

        pipeline.add(memory).unwrap();

        // Delete the memory
        let deleted = pipeline.delete(&id).unwrap();
        assert!(deleted);

        // Verify it's removed from storage
        let retrieved = pipeline.storage.get_memory(&id).unwrap();
        assert!(retrieved.is_none());

        // Verify it's removed from vector index (search won't find it)
        {
            let index = pipeline.vector_index.read().unwrap();
            let results = index.search(&vec![0.1; 384], 10).unwrap();
            assert!(results.is_empty() || !results.iter().any(|r| r.id == id));
        }

        // Verify it's removed from entity graph
        let graph = pipeline.graph_manager.read().unwrap();
        let entities = graph.get_memory_entities(&id);
        assert!(entities.is_empty());
    }

    #[test]
    fn test_pipeline_delete_cleans_orphaned_entities() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add memory with entity
        let memory = Memory::new("Alice is here".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();
        pipeline.add(memory).unwrap();

        // Verify entity exists
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());
        let alice = alice.unwrap();
        assert_eq!(alice.mention_count, 1);

        // Delete the memory
        pipeline.delete(&id).unwrap();

        // Verify entity is removed (orphaned)
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_none());
    }

    #[test]
    fn test_pipeline_delete_keeps_non_orphaned_entities() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add two memories mentioning Alice
        let memory1 = Memory::new("Alice is here".to_string(), vec![0.1; 384]);
        let id1 = memory1.id.clone();
        pipeline.add(memory1).unwrap();

        let memory2 = Memory::new("Alice is there too".to_string(), vec![0.2; 384]);
        let id2 = memory2.id.clone();
        pipeline.add(memory2).unwrap();

        // Verify Alice has 2 mentions
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());
        let alice = alice.unwrap();
        assert_eq!(alice.mention_count, 2);

        // Delete first memory
        pipeline.delete(&id1).unwrap();

        // Verify Alice still exists with 1 mention
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());
        let alice = alice.unwrap();
        assert_eq!(alice.mention_count, 1);

        // Delete second memory
        pipeline.delete(&id2).unwrap();

        // Now Alice should be removed
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_none());
    }

    #[test]
    fn test_pipeline_delete_nonexistent() {
        let (pipeline, _dir) = create_test_pipeline();

        let id = MemoryId::new();
        let deleted = pipeline.delete(&id).unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_pipeline_rollback_on_vector_index_failure() {
        let (pipeline, _dir) = create_test_pipeline();

        // Create memory with wrong dimension to trigger vector index error
        let memory = Memory::new("Test".to_string(), vec![0.1; 512]);
        let id = memory.id.clone();

        let result = pipeline.add(memory);
        assert!(result.is_err());

        // Verify memory was NOT stored (rollback worked)
        let retrieved = pipeline.storage.get_memory(&id).unwrap();
        assert!(retrieved.is_none());
    }

    // Batch operation tests
    #[test]
    fn test_pipeline_add_batch_success() {
        let (pipeline, _dir) = create_test_pipeline();

        // Create batch inputs
        let inputs = vec![
            MemoryInput::new("Memory 1".to_string(), vec![0.1; 384]),
            MemoryInput::new("Memory 2".to_string(), vec![0.2; 384]),
            MemoryInput::new("Memory 3".to_string(), vec![0.3; 384]),
        ];

        // Add batch
        let result = pipeline.add_batch(inputs, None).unwrap();

        // Verify results
        assert_eq!(result.created_count, 3);
        assert_eq!(result.ids.len(), 3);
        assert_eq!(result.errors.len(), 0);
        assert!(result.is_success());

        // Verify all memories were stored
        for id in &result.ids {
            let memory = pipeline.storage.get_memory(id).unwrap();
            assert!(memory.is_some());
        }
    }

    #[test]
    fn test_pipeline_add_batch_with_metadata_and_source() {
        let (pipeline, _dir) = create_test_pipeline();

        // Create inputs with metadata and source
        use crate::types::{Source, SourceType};
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("project".to_string(), "alpha".to_string());

        let source = Source::new(SourceType::Manual).with_id("batch_test");

        let input = MemoryInput::new("Test with metadata".to_string(), vec![0.1; 384])
            .with_metadata(metadata.clone())
            .with_source(source);

        // Add batch
        let result = pipeline.add_batch(vec![input], None).unwrap();

        assert_eq!(result.created_count, 1);

        // Verify metadata and source
        let memory = pipeline
            .storage
            .get_memory(&result.ids[0])
            .unwrap()
            .unwrap();
        assert_eq!(memory.get_metadata("project"), Some(&"alpha".to_string()));
        assert!(memory.get_source().unwrap().is_some());
    }

    #[test]
    fn test_pipeline_add_batch_partial_failure() {
        let (pipeline, _dir) = create_test_pipeline();

        // Create batch with one invalid embedding
        let inputs = vec![
            MemoryInput::new("Memory 1".to_string(), vec![0.1; 384]),
            MemoryInput::new("Memory 2".to_string(), vec![0.2; 512]), // Wrong dimension
            MemoryInput::new("Memory 3".to_string(), vec![0.3; 384]),
        ];

        // Add batch
        let result = pipeline.add_batch(inputs, None).unwrap();

        // Verify partial success
        assert_eq!(result.created_count, 2); // Only 1 and 3 succeeded
        assert_eq!(result.ids.len(), 2);
        assert_eq!(result.errors.len(), 1);
        assert!(!result.is_success());
        assert!(result.has_errors());

        // Verify error details
        assert_eq!(result.errors[0].index, 1); // Second memory failed
        assert!(result.errors[0].message.contains("Vector index failed"));
    }

    #[test]
    fn test_pipeline_add_batch_empty() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add empty batch
        let result = pipeline.add_batch(vec![], None).unwrap();

        assert_eq!(result.created_count, 0);
        assert_eq!(result.ids.len(), 0);
        assert_eq!(result.errors.len(), 0);
        assert!(result.is_success());
    }

    #[test]
    fn test_pipeline_delete_batch_success() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add three memories
        let memory1 = Memory::new("Memory 1".to_string(), vec![0.1; 384]);
        let id1 = memory1.id.clone();
        pipeline.add(memory1).unwrap();

        let memory2 = Memory::new("Memory 2".to_string(), vec![0.2; 384]);
        let id2 = memory2.id.clone();
        pipeline.add(memory2).unwrap();

        let memory3 = Memory::new("Memory 3".to_string(), vec![0.3; 384]);
        let id3 = memory3.id.clone();
        pipeline.add(memory3).unwrap();

        // Delete batch
        let deleted_count = pipeline
            .delete_batch(vec![id1.clone(), id2.clone(), id3.clone()])
            .unwrap();

        assert_eq!(deleted_count, 3);

        // Verify all are deleted
        assert!(pipeline.storage.get_memory(&id1).unwrap().is_none());
        assert!(pipeline.storage.get_memory(&id2).unwrap().is_none());
        assert!(pipeline.storage.get_memory(&id3).unwrap().is_none());
    }

    #[test]
    fn test_pipeline_delete_batch_partial() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add two memories
        let memory1 = Memory::new("Memory 1".to_string(), vec![0.1; 384]);
        let id1 = memory1.id.clone();
        pipeline.add(memory1).unwrap();

        let memory2 = Memory::new("Memory 2".to_string(), vec![0.2; 384]);
        let id2 = memory2.id.clone();
        pipeline.add(memory2).unwrap();

        // Try to delete 3 (one doesn't exist)
        let fake_id = MemoryId::new();
        let deleted_count = pipeline
            .delete_batch(vec![id1.clone(), fake_id, id2.clone()])
            .unwrap();

        // Only 2 should be deleted
        assert_eq!(deleted_count, 2);

        assert!(pipeline.storage.get_memory(&id1).unwrap().is_none());
        assert!(pipeline.storage.get_memory(&id2).unwrap().is_none());
    }

    #[test]
    fn test_pipeline_delete_batch_with_entities() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add memories with entities
        let memory1 = Memory::new("Alice and Bob are here".to_string(), vec![0.1; 384]);
        let id1 = memory1.id.clone();
        pipeline.add(memory1).unwrap();

        let memory2 = Memory::new("Alice went to the store".to_string(), vec![0.2; 384]);
        let id2 = memory2.id.clone();
        pipeline.add(memory2).unwrap();

        // Verify Alice has 2 mentions
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());
        assert_eq!(alice.unwrap().mention_count, 2);

        // Delete batch
        let deleted_count = pipeline.delete_batch(vec![id1, id2]).unwrap();
        assert_eq!(deleted_count, 2);

        // Verify Alice is now orphaned and removed
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_none());

        // Verify Bob is also removed
        let bob = pipeline.storage.find_entity_by_name("Bob").unwrap();
        assert!(bob.is_none());
    }

    #[test]
    fn test_pipeline_delete_batch_empty() {
        let (pipeline, _dir) = create_test_pipeline();

        // Delete empty batch
        let deleted_count = pipeline.delete_batch(vec![]).unwrap();
        assert_eq!(deleted_count, 0);
    }

    // Deduplication tests
    #[test]
    fn test_pipeline_add_with_dedup_creates_first() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("test content".to_string(), vec![0.1; 384]);
        let result = pipeline.add_with_dedup(memory).unwrap();

        assert!(result.created);
        assert!(result.is_created());
        assert!(!result.is_duplicate());
        assert!(result.existing_id.is_none());
    }

    #[test]
    fn test_pipeline_add_with_dedup_detects_duplicate() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add first memory
        let memory1 = Memory::new("duplicate content".to_string(), vec![0.1; 384]);
        let result1 = pipeline.add_with_dedup(memory1).unwrap();
        assert!(result1.created);

        // Try to add same content again
        let memory2 = Memory::new("duplicate content".to_string(), vec![0.2; 384]);
        let result2 = pipeline.add_with_dedup(memory2).unwrap();

        // Should detect duplicate
        assert!(!result2.created);
        assert!(result2.is_duplicate());
        assert_eq!(result2.id, result1.id);
        assert_eq!(result2.existing_id, Some(result1.id));
    }

    #[test]
    fn test_pipeline_add_with_dedup_different_content() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory1 = Memory::new("content 1".to_string(), vec![0.1; 384]);
        let result1 = pipeline.add_with_dedup(memory1).unwrap();

        let memory2 = Memory::new("content 2".to_string(), vec![0.2; 384]);
        let result2 = pipeline.add_with_dedup(memory2).unwrap();

        // Different content should create new memory
        assert!(result1.created);
        assert!(result2.created);
        assert_ne!(result1.id, result2.id);
    }

    #[test]
    fn test_pipeline_add_with_dedup_whitespace_sensitive() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory1 = Memory::new("test content".to_string(), vec![0.1; 384]);
        let result1 = pipeline.add_with_dedup(memory1).unwrap();

        // Different whitespace = different content (no normalization in add_with_dedup)
        let memory2 = Memory::new("test  content".to_string(), vec![0.2; 384]);
        let result2 = pipeline.add_with_dedup(memory2).unwrap();

        // Should create new memory (different content)
        assert!(result1.created);
        assert!(result2.created);
        assert_ne!(result1.id, result2.id);
    }

    // Upsert tests
    #[test]
    fn test_pipeline_upsert_creates_new() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("initial content".to_string(), vec![0.1; 384]);
        let result = pipeline.upsert("key1", memory).unwrap();

        assert!(result.created);
        assert!(result.is_created());
        assert!(!result.updated);
        assert!(result.previous_content.is_none());
    }

    #[test]
    fn test_pipeline_upsert_updates_existing() {
        let (pipeline, _dir) = create_test_pipeline();

        // First upsert - creates
        let memory1 = Memory::new("original content".to_string(), vec![0.1; 384]);
        let result1 = pipeline.upsert("key1", memory1).unwrap();
        assert!(result1.created);
        let first_id = result1.id.clone();

        // Second upsert - updates
        let memory2 = Memory::new("updated content".to_string(), vec![0.2; 384]);
        let result2 = pipeline.upsert("key1", memory2).unwrap();

        assert!(!result2.created);
        assert!(result2.updated);
        assert!(result2.is_updated());
        assert_eq!(
            result2.previous_content,
            Some("original content".to_string())
        );

        // Original memory should be deleted
        let original = pipeline.storage.get_memory(&first_id).unwrap();
        assert!(original.is_none());

        // New memory should exist
        let updated = pipeline.storage.get_memory(&result2.id).unwrap();
        assert!(updated.is_some());
        assert_eq!(updated.unwrap().content, "updated content");
    }

    #[test]
    fn test_pipeline_upsert_different_keys() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory1 = Memory::new("content 1".to_string(), vec![0.1; 384]);
        let result1 = pipeline.upsert("key1", memory1).unwrap();

        let memory2 = Memory::new("content 2".to_string(), vec![0.2; 384]);
        let result2 = pipeline.upsert("key2", memory2).unwrap();

        // Different keys should create different memories
        assert!(result1.created);
        assert!(result2.created);
        assert_ne!(result1.id, result2.id);

        // Both should exist
        assert!(pipeline.storage.get_memory(&result1.id).unwrap().is_some());
        assert!(pipeline.storage.get_memory(&result2.id).unwrap().is_some());
    }

    #[test]
    fn test_pipeline_upsert_updates_vector_index() {
        let (pipeline, _dir) = create_test_pipeline();

        // First upsert
        let memory1 = Memory::new("test".to_string(), vec![0.1; 384]);
        let result1 = pipeline.upsert("key1", memory1).unwrap();

        // Second upsert with different embedding
        let memory2 = Memory::new("test updated".to_string(), vec![0.9; 384]);
        let result2 = pipeline.upsert("key1", memory2).unwrap();

        // Verify the logical key points to the new memory
        let found_id = pipeline.storage.find_by_logical_key("key1").unwrap();
        assert_eq!(found_id, Some(result2.id.clone()));

        // Verify old memory is deleted
        assert!(pipeline.storage.get_memory(&result1.id).unwrap().is_none());

        // Verify new memory exists
        let new_memory = pipeline.storage.get_memory(&result2.id).unwrap();
        assert!(new_memory.is_some());
        assert_eq!(new_memory.unwrap().content, "test updated");

        // Verify vector index has results (search should work)
        {
            let index = pipeline.vector_index.read().unwrap();
            let results = index.search(&vec![0.9; 384], 5).unwrap();
            assert!(
                !results.is_empty(),
                "Vector index should have results after upsert"
            );
        }
    }

    #[test]
    fn test_pipeline_upsert_cleans_up_entities() {
        let (pipeline, _dir) = create_test_pipeline();

        // First upsert with entity
        let memory1 = Memory::new("Alice is here".to_string(), vec![0.1; 384]);
        let result1 = pipeline.upsert("key1", memory1).unwrap();

        // Verify Alice exists
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());

        // Second upsert with different entity
        let memory2 = Memory::new("Bob is here".to_string(), vec![0.2; 384]);
        let result2 = pipeline.upsert("key1", memory2).unwrap();

        // Alice should be removed (orphaned)
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_none());

        // Bob should exist
        let bob = pipeline.storage.find_entity_by_name("Bob").unwrap();
        assert!(bob.is_some());
    }

    // ========== Entity Profile Cleanup Tests ==========

    #[test]
    fn test_pipeline_delete_cleans_entity_profiles() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add a memory
        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();
        pipeline.add(memory).unwrap();

        // Manually create an entity profile with a fact from this memory
        let mut profile = crate::types::EntityProfile::new(
            crate::types::EntityId::new(),
            "TestEntity".to_string(),
            "concept".to_string(),
        );
        profile.add_fact(crate::types::EntityFact::new(
            "test_fact",
            "test_value",
            0.9,
            id.clone(),
        ));
        profile.add_source_memory(id.clone());
        pipeline.storage.store_entity_profile(&profile).unwrap();

        // Verify profile exists with the fact
        let stored = pipeline.storage.get_entity_profile("TestEntity").unwrap().unwrap();
        assert_eq!(stored.total_facts(), 1);
        assert!(stored.source_memories.contains(&id));

        // Delete the memory
        let deleted = pipeline.delete(&id).unwrap();
        assert!(deleted);

        // Profile should now be empty and deleted
        let profile_after = pipeline.storage.get_entity_profile("TestEntity").unwrap();
        assert!(profile_after.is_none(), "Empty profile should be deleted");
    }

    #[test]
    fn test_pipeline_delete_preserves_other_facts_in_profile() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add two memories
        let memory1 = Memory::new("Memory 1".to_string(), vec![0.1; 384]);
        let id1 = memory1.id.clone();
        pipeline.add(memory1).unwrap();

        let memory2 = Memory::new("Memory 2".to_string(), vec![0.2; 384]);
        let id2 = memory2.id.clone();
        pipeline.add(memory2).unwrap();

        // Create a profile with facts from both memories
        let mut profile = crate::types::EntityProfile::new(
            crate::types::EntityId::new(),
            "SharedEntity".to_string(),
            "concept".to_string(),
        );
        profile.add_fact(crate::types::EntityFact::new(
            "fact_from_mem1",
            "value1",
            0.9,
            id1.clone(),
        ));
        profile.add_fact(crate::types::EntityFact::new(
            "fact_from_mem2",
            "value2",
            0.85,
            id2.clone(),
        ));
        profile.add_source_memory(id1.clone());
        profile.add_source_memory(id2.clone());
        pipeline.storage.store_entity_profile(&profile).unwrap();

        // Verify profile has both facts
        let stored = pipeline.storage.get_entity_profile("SharedEntity").unwrap().unwrap();
        assert_eq!(stored.total_facts(), 2);

        // Delete memory1
        pipeline.delete(&id1).unwrap();

        // Profile should still exist with only mem2's fact
        let profile_after = pipeline.storage.get_entity_profile("SharedEntity").unwrap().unwrap();
        assert_eq!(profile_after.total_facts(), 1);
        assert!(!profile_after.source_memories.contains(&id1));
        assert!(profile_after.source_memories.contains(&id2));

        // The remaining fact should be from memory2
        let remaining_facts = profile_after.get_facts("fact_from_mem2");
        assert_eq!(remaining_facts.len(), 1);
        assert_eq!(remaining_facts[0].value, "value2");
    }

    #[test]
    fn test_pipeline_delete_nonexistent_memory_doesnt_affect_profiles() {
        let (pipeline, _dir) = create_test_pipeline();

        // Create a profile with some facts
        let memory_id = MemoryId::new();
        let mut profile = crate::types::EntityProfile::new(
            crate::types::EntityId::new(),
            "PersistentEntity".to_string(),
            "concept".to_string(),
        );
        profile.add_fact(crate::types::EntityFact::new(
            "test_fact",
            "test_value",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id);
        pipeline.storage.store_entity_profile(&profile).unwrap();

        // Try to delete a nonexistent memory
        let fake_id = MemoryId::new();
        let deleted = pipeline.delete(&fake_id).unwrap();
        assert!(!deleted);

        // Profile should be unchanged
        let profile_after = pipeline.storage.get_entity_profile("PersistentEntity").unwrap().unwrap();
        assert_eq!(profile_after.total_facts(), 1);
    }

    #[test]
    fn test_pipeline_delete_batch_cleans_profiles() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add three memories
        let memory1 = Memory::new("Memory 1".to_string(), vec![0.1; 384]);
        let id1 = memory1.id.clone();
        pipeline.add(memory1).unwrap();

        let memory2 = Memory::new("Memory 2".to_string(), vec![0.2; 384]);
        let id2 = memory2.id.clone();
        pipeline.add(memory2).unwrap();

        let memory3 = Memory::new("Memory 3".to_string(), vec![0.3; 384]);
        let id3 = memory3.id.clone();
        pipeline.add(memory3).unwrap();

        // Create profiles with facts from different memories
        let mut profile1 = crate::types::EntityProfile::new(
            crate::types::EntityId::new(),
            "Entity1".to_string(),
            "concept".to_string(),
        );
        profile1.add_fact(crate::types::EntityFact::new(
            "fact1",
            "value1",
            0.9,
            id1.clone(),
        ));
        profile1.add_source_memory(id1.clone());
        pipeline.storage.store_entity_profile(&profile1).unwrap();

        let mut profile2 = crate::types::EntityProfile::new(
            crate::types::EntityId::new(),
            "Entity2".to_string(),
            "concept".to_string(),
        );
        profile2.add_fact(crate::types::EntityFact::new(
            "fact2",
            "value2",
            0.85,
            id2.clone(),
        ));
        profile2.add_fact(crate::types::EntityFact::new(
            "fact3",
            "value3",
            0.8,
            id3.clone(),
        ));
        profile2.add_source_memory(id2.clone());
        profile2.add_source_memory(id3.clone());
        pipeline.storage.store_entity_profile(&profile2).unwrap();

        // Delete memories 1 and 2
        let deleted = pipeline.delete_batch(vec![id1.clone(), id2.clone()]).unwrap();
        assert_eq!(deleted, 2);

        // Entity1 profile should be deleted (only had facts from id1)
        assert!(pipeline.storage.get_entity_profile("Entity1").unwrap().is_none());

        // Entity2 profile should exist with only id3's fact
        let profile2_after = pipeline.storage.get_entity_profile("Entity2").unwrap().unwrap();
        assert_eq!(profile2_after.total_facts(), 1);
        assert!(profile2_after.source_memories.contains(&id3));
        assert!(!profile2_after.source_memories.contains(&id1));
        assert!(!profile2_after.source_memories.contains(&id2));
    }

    // ========== Entity-Only Source Memory Linking Tests ==========

    #[cfg(feature = "entity-extraction")]
    #[test]
    fn test_entity_only_linking_creates_profile_and_links_source_memory() {
        use crate::extraction::{ExtractedEntity, ExtractionResult};

        let (pipeline, _dir) = create_test_pipeline();

        // Add a memory to storage first
        let memory = Memory::new("Caroline went to the park".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();
        pipeline.storage.store_memory(&memory).unwrap();

        // Create extraction with entity but NO entity_facts
        let extraction = ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "Caroline".to_string(),
                entity_type: "person".to_string(),
            }],
            topics: vec!["park".to_string()],
            ..Default::default()
        };

        // Call the function under test
        pipeline
            .update_entity_profiles_from_llm(&id, &extraction)
            .unwrap();

        // Profile should exist with source_memory linked, even without facts
        let profile = pipeline
            .storage
            .get_entity_profile("Caroline")
            .unwrap()
            .expect("Profile should be created for entity-only extraction");
        assert!(
            profile.source_memories.contains(&id),
            "source_memory should be linked even without entity_facts"
        );
        assert_eq!(profile.total_facts(), 0, "No facts should be added");
    }

    #[cfg(feature = "entity-extraction")]
    #[test]
    fn test_entity_only_linking_skips_already_linked_entities() {
        use crate::extraction::{ExtractedEntity, ExtractedFact, ExtractionResult};

        let (pipeline, _dir) = create_test_pipeline();

        // Add a memory
        let memory = Memory::new("Caroline is a counselor".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();
        pipeline.storage.store_memory(&memory).unwrap();

        // Extraction with BOTH entity and entity_facts for Caroline
        let extraction = ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "Caroline".to_string(),
                entity_type: "person".to_string(),
            }],
            entity_facts: vec![ExtractedFact {
                entity: "Caroline".to_string(),
                fact_type: "occupation".to_string(),
                value: "counselor".to_string(),
                confidence: 0.9,
            }],
            ..Default::default()
        };

        pipeline
            .update_entity_profiles_from_llm(&id, &extraction)
            .unwrap();

        // Profile should have the fact AND source_memory (from entity_facts path)
        let profile = pipeline
            .storage
            .get_entity_profile("Caroline")
            .unwrap()
            .unwrap();
        assert_eq!(profile.total_facts(), 1);
        assert!(profile.source_memories.contains(&id));
    }

    #[cfg(feature = "entity-extraction")]
    #[test]
    fn test_entity_only_linking_mixed_entities() {
        use crate::extraction::{ExtractedEntity, ExtractedFact, ExtractionResult};

        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new(
            "Caroline and Bob discussed therapy".to_string(),
            vec![0.1; 384],
        );
        let id = memory.id.clone();
        pipeline.storage.store_memory(&memory).unwrap();

        // Caroline has facts, Bob does not
        let extraction = ExtractionResult {
            entities: vec![
                ExtractedEntity {
                    name: "Caroline".to_string(),
                    entity_type: "person".to_string(),
                },
                ExtractedEntity {
                    name: "Bob".to_string(),
                    entity_type: "person".to_string(),
                },
            ],
            entity_facts: vec![ExtractedFact {
                entity: "Caroline".to_string(),
                fact_type: "interest".to_string(),
                value: "therapy".to_string(),
                confidence: 0.8,
            }],
            ..Default::default()
        };

        pipeline
            .update_entity_profiles_from_llm(&id, &extraction)
            .unwrap();

        // Caroline: has fact + source_memory (via entity_facts path)
        let caroline = pipeline
            .storage
            .get_entity_profile("Caroline")
            .unwrap()
            .unwrap();
        assert_eq!(caroline.total_facts(), 1);
        assert!(caroline.source_memories.contains(&id));

        // Bob: no facts but still has source_memory (via entity-only path)
        let bob = pipeline
            .storage
            .get_entity_profile("Bob")
            .unwrap()
            .expect("Bob's profile should be created via entity-only linking");
        assert_eq!(bob.total_facts(), 0);
        assert!(
            bob.source_memories.contains(&id),
            "Bob should have source_memory linked even without facts"
        );
    }

    #[cfg(feature = "entity-extraction")]
    #[test]
    fn test_extraction_time_alias_resolution() {
        use crate::extraction::{ExtractedEntity, ExtractedFact, ExtractionResult};
        use crate::types::{EntityId, EntityProfile};

        let (pipeline, _dir) = create_test_pipeline();

        // Pre-create "melanie" profile (canonical form)
        let melanie_profile = EntityProfile::new(
            EntityId::new(),
            "melanie".to_string(),
            "person".to_string(),
        );
        pipeline
            .storage
            .store_entity_profile(&melanie_profile)
            .unwrap();

        let memory = Memory::new(
            "Mel loves hiking and cooking".to_string(),
            vec![0.1; 384],
        );
        let id = memory.id.clone();
        pipeline.storage.store_memory(&memory).unwrap();

        // Extraction uses raw name "Mel"
        let extraction = ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "Mel".to_string(),
                entity_type: "person".to_string(),
            }],
            entity_facts: vec![ExtractedFact {
                entity: "Mel".to_string(),
                fact_type: "hobby".to_string(),
                value: "hiking".to_string(),
                confidence: 0.9,
            }],
            ..Default::default()
        };

        pipeline
            .update_entity_profiles_from_llm(&id, &extraction)
            .unwrap();

        // "Mel" should be resolved to "melanie" — fact stored under canonical profile
        let melanie = pipeline
            .storage
            .get_entity_profile("melanie")
            .unwrap()
            .expect("melanie profile should exist");
        assert_eq!(melanie.total_facts(), 1, "Fact should be stored under 'melanie'");
        assert!(melanie.source_memories.contains(&id));

        // No separate "mel" profile should be created
        let mel = pipeline.storage.get_entity_profile("mel").unwrap();
        // mel profile exists (pre-existing from storage key matching) but should have no NEW facts
        // Actually the canonical form takes over — facts go to melanie
        assert!(
            mel.is_none() || mel.as_ref().unwrap().total_facts() == 0,
            "No facts should be stored under raw 'mel' name"
        );
    }

    #[cfg(feature = "entity-extraction")]
    #[test]
    fn test_entity_type_filter_blocks_non_person_profiles() {
        use crate::extraction::{ExtractedEntity, ExtractedFact, ExtractionResult};

        let (mut pipeline, _dir) = create_test_pipeline();
        // Default allows person/organization/location
        assert_eq!(pipeline.profile_entity_types.len(), 3);

        let memory = Memory::new(
            "Dogs played basketball in nature".to_string(),
            vec![0.1; 384],
        );
        let id = memory.id.clone();
        pipeline.storage.store_memory(&memory).unwrap();

        let extraction = ExtractionResult {
            entities: vec![
                ExtractedEntity {
                    name: "Caroline".to_string(),
                    entity_type: "person".to_string(),
                },
                ExtractedEntity {
                    name: "dogs".to_string(),
                    entity_type: "animal".to_string(),
                },
                ExtractedEntity {
                    name: "basketball".to_string(),
                    entity_type: "activity".to_string(),
                },
            ],
            entity_facts: vec![
                ExtractedFact {
                    entity: "Caroline".to_string(),
                    fact_type: "hobby".to_string(),
                    value: "basketball".to_string(),
                    confidence: 0.8,
                },
                ExtractedFact {
                    entity: "dogs".to_string(),
                    fact_type: "activity".to_string(),
                    value: "playing".to_string(),
                    confidence: 0.7,
                },
            ],
            ..Default::default()
        };

        pipeline
            .update_entity_profiles_from_llm(&id, &extraction)
            .unwrap();

        // Person entity should have profile
        let caroline = pipeline.storage.get_entity_profile("Caroline").unwrap();
        assert!(caroline.is_some(), "Person entity should have profile");
        assert_eq!(caroline.unwrap().total_facts(), 1);

        // Non-person entities should NOT have profiles
        let dogs = pipeline.storage.get_entity_profile("dogs").unwrap();
        assert!(dogs.is_none(), "Animal entity should be filtered out");

        let basketball = pipeline.storage.get_entity_profile("basketball").unwrap();
        assert!(basketball.is_none(), "Activity entity should be filtered out");
    }

    #[cfg(feature = "entity-extraction")]
    #[test]
    fn test_entity_type_filter_empty_allows_all() {
        use crate::extraction::{ExtractedEntity, ExtractedFact, ExtractionResult};

        let (mut pipeline, _dir) = create_test_pipeline();
        // Clear the filter to allow all types
        pipeline.set_profile_entity_types(Vec::new());

        let memory = Memory::new("Dogs are pets".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();
        pipeline.storage.store_memory(&memory).unwrap();

        let extraction = ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "dogs".to_string(),
                entity_type: "animal".to_string(),
            }],
            entity_facts: vec![ExtractedFact {
                entity: "dogs".to_string(),
                fact_type: "category".to_string(),
                value: "pet".to_string(),
                confidence: 0.9,
            }],
            ..Default::default()
        };

        pipeline
            .update_entity_profiles_from_llm(&id, &extraction)
            .unwrap();

        // With empty filter, all types should be allowed
        let dogs = pipeline.storage.get_entity_profile("dogs").unwrap();
        assert!(dogs.is_some(), "Empty filter should allow all entity types");
    }

    #[cfg(feature = "entity-extraction")]
    #[test]
    fn test_ingestion_alias_ordering() {
        // Bug 1: If "Mel" is processed before "Melanie" exists, known_names
        // won't contain "melanie" yet. With mutable known_names, "melanie" gets
        // pushed after first profile creation, so the second extraction resolves.
        use crate::extraction::{ExtractedEntity, ExtractedFact, ExtractionResult};
        use crate::types::{EntityId, EntityProfile};

        let (pipeline, _dir) = create_test_pipeline();

        let mem1 = Memory::new("Mel loves hiking".to_string(), vec![0.1; 384]);
        let id1 = mem1.id.clone();
        pipeline.storage.store_memory(&mem1).unwrap();

        let mem2 = Memory::new("Melanie plays guitar".to_string(), vec![0.2; 384]);
        let id2 = mem2.id.clone();
        pipeline.storage.store_memory(&mem2).unwrap();

        // First extraction: "Mel" — no canonical exists yet, creates "mel" profile
        let extraction1 = ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "Mel".to_string(),
                entity_type: "person".to_string(),
            }],
            entity_facts: vec![ExtractedFact {
                entity: "Mel".to_string(),
                fact_type: "hobby".to_string(),
                value: "hiking".to_string(),
                confidence: 0.9,
            }],
            ..Default::default()
        };
        pipeline
            .update_entity_profiles_from_llm(&id1, &extraction1)
            .unwrap();

        // Second extraction: "Melanie" — should create "melanie" profile
        let extraction2 = ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "Melanie".to_string(),
                entity_type: "person".to_string(),
            }],
            entity_facts: vec![ExtractedFact {
                entity: "Melanie".to_string(),
                fact_type: "instrument".to_string(),
                value: "guitar".to_string(),
                confidence: 0.9,
            }],
            ..Default::default()
        };
        pipeline
            .update_entity_profiles_from_llm(&id2, &extraction2)
            .unwrap();

        // Now process a THIRD extraction with "Mel" again — should resolve to "melanie"
        // because known_names was updated with "melanie" after extraction2
        let mem3 = Memory::new("Mel went to the park".to_string(), vec![0.3; 384]);
        let id3 = mem3.id.clone();
        pipeline.storage.store_memory(&mem3).unwrap();

        let extraction3 = ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "Mel".to_string(),
                entity_type: "person".to_string(),
            }],
            entity_facts: vec![ExtractedFact {
                entity: "Mel".to_string(),
                fact_type: "activity".to_string(),
                value: "went to the park".to_string(),
                confidence: 0.9,
            }],
            ..Default::default()
        };
        pipeline
            .update_entity_profiles_from_llm(&id3, &extraction3)
            .unwrap();

        // "melanie" profile should have the guitar fact + the park activity
        let melanie = pipeline
            .storage
            .get_entity_profile("melanie")
            .unwrap()
            .expect("melanie profile should exist");
        assert!(
            melanie.total_facts() >= 2,
            "melanie should have guitar + park facts, got {}",
            melanie.total_facts()
        );
        assert!(
            melanie.source_memories.contains(&id2),
            "melanie should have source memory from extraction2"
        );
        assert!(
            melanie.source_memories.contains(&id3),
            "melanie should have source memory from extraction3 (resolved from Mel)"
        );
    }
}
