//! Query planner for coordinating multi-dimensional queries
//!
//! The QueryPlanner classifies query intent and coordinates retrieval across
//! semantic, temporal, causal, and entity dimensions with adaptive weighting.

use crate::{
    error::Result,
    graph::GraphManager,
    index::{TemporalIndex, VectorIndex},
    ingest::{get_causal_extractor, SlmMetadata},
    query::{
        aggregator::MultiTurnAggregator,
        fusion::{FusedResult, FusionEngine},
        intent::{IntentClassification, IntentClassifier, QueryIntent},
        profile_search::ProfileSearch,
    },
    storage::StorageEngine,
    types::{Memory, MemoryId, MetadataFilter, Timestamp},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Query planner that coordinates multi-dimensional retrieval
pub struct QueryPlanner {
    pub(crate) storage: Arc<StorageEngine>,
    pub(crate) vector_index: Arc<RwLock<VectorIndex>>,
    pub(crate) bm25_index: Arc<crate::index::BM25Index>,
    pub(crate) temporal_index: Arc<TemporalIndex>,
    pub(crate) graph_manager: Arc<RwLock<GraphManager>>,
    intent_classifier: IntentClassifier,
    #[cfg(feature = "slm")]
    slm_classifier: Option<Arc<std::sync::Mutex<crate::slm::SlmClassifier>>>,
    /// Whether to use SLM for query classification (default: false)
    /// When false, uses pattern-based classification for fast queries
    slm_query_classification_enabled: bool,
    fusion_engine: FusionEngine,
    semantic_prefilter_threshold: f32,
    /// Profile-based search for entity fact lookup
    profile_search: ProfileSearch,
    /// Adaptive-K (Top-p) threshold for dynamic result count selection.
    /// 0.0 = disabled (always return limit), 0.7 = recommended.
    adaptive_k_threshold: f32,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new(
        storage: Arc<StorageEngine>,
        vector_index: Arc<RwLock<VectorIndex>>,
        bm25_index: Arc<crate::index::BM25Index>,
        temporal_index: Arc<TemporalIndex>,
        graph_manager: Arc<RwLock<GraphManager>>,
        fusion_semantic_threshold: f32,
        semantic_prefilter_threshold: f32,
        fusion_strategy: crate::query::FusionStrategy,
        rrf_k: f32,
        #[cfg(feature = "slm")]
        slm_config: Option<crate::slm::SlmConfig>,
        slm_query_classification_enabled: bool,
        adaptive_k_threshold: f32,
    ) -> Result<Self> {
        // Initialize SLM classifier if configured
        #[cfg(feature = "slm")]
        let slm_classifier = if let Some(config) = slm_config {
            eprintln!("[DEBUG-QP] SLM config received! model_id: {}", config.model_id);
            eprintln!("[DEBUG-QP] model_path: {:?}", config.model_path);
            tracing::info!("Initializing SLM classifier with model: {}", config.model_id);
            match crate::slm::SlmClassifier::new(config) {
                Ok(classifier) => {
                    eprintln!("[DEBUG-QP] ✓ SLM classifier initialized successfully");
                    tracing::info!("SLM classifier initialized successfully (lazy loading)");
                    Some(Arc::new(std::sync::Mutex::new(classifier)))
                }
                Err(e) => {
                    eprintln!("[DEBUG-QP] ✗ SLM classifier initialization FAILED: {}", e);
                    tracing::warn!("Failed to initialize SLM classifier: {}, falling back to patterns", e);
                    None
                }
            }
        } else {
            eprintln!("[DEBUG-QP] No SLM config provided to QueryPlanner");
            None
        };

        // Create profile search engine
        let profile_search = ProfileSearch::new(Arc::clone(&storage));

        Ok(Self {
            storage,
            vector_index,
            bm25_index,
            temporal_index,
            graph_manager,
            intent_classifier: IntentClassifier::new(),
            #[cfg(feature = "slm")]
            slm_classifier,
            slm_query_classification_enabled,
            fusion_engine: FusionEngine::new()
                .with_semantic_threshold(fusion_semantic_threshold)
                .with_strategy(fusion_strategy)
                .with_rrf_k(rrf_k),
            semantic_prefilter_threshold,
            profile_search,
            adaptive_k_threshold,
        })
    }

    /// Classify query intent using SLM (if enabled and available) or patterns
    ///
    /// If slm_query_classification_enabled is true and SLM is available, tries SLM
    /// classification first. Otherwise, or on any error, falls back to pattern-based
    /// classification for fast queries.
    ///
    /// # Arguments
    ///
    /// * `query_text` - The natural language query
    ///
    /// # Returns
    ///
    /// Intent classification with confidence score
    fn classify_intent(&self, query_text: &str) -> Result<IntentClassification> {
        eprintln!("[DEBUG-QP] classify_intent() called for: '{}'", query_text);

        // Only use SLM classification if explicitly enabled (default: false)
        // This is intentionally disabled by default for fast queries
        #[cfg(feature = "slm")]
        if self.slm_query_classification_enabled {
            if let Some(slm_classifier) = &self.slm_classifier {
                eprintln!("[DEBUG-QP] SLM query classification ENABLED, attempting to use it");
                // Try SLM classification
                match slm_classifier.lock() {
                    Ok(mut classifier) => {
                        eprintln!("[DEBUG-QP] Acquired SLM classifier lock, calling classify_intent");
                        match classifier.classify_intent(query_text) {
                            Ok(classification) => {
                                eprintln!("[DEBUG-QP] ✓ SLM classification succeeded: {:?} (confidence: {:.2})",
                                    classification.intent, classification.confidence);
                                tracing::debug!(
                                    "SLM classified query as {:?} (confidence: {:.2})",
                                    classification.intent,
                                    classification.confidence
                                );
                                return Ok(classification);
                            }
                            Err(e) => {
                                eprintln!("[DEBUG-QP] ✗ SLM classification failed: {}, falling back", e);
                                tracing::warn!(
                                    "SLM classification failed: {}, falling back to patterns",
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[DEBUG-QP] ✗ Failed to acquire SLM classifier lock: {}", e);
                        tracing::warn!(
                            "Failed to acquire SLM classifier lock: {}, falling back to patterns",
                            e
                        );
                    }
                }
            } else {
                eprintln!("[DEBUG-QP] SLM classifier is NOT available");
            }
        } else {
            eprintln!("[DEBUG-QP] SLM query classification DISABLED (using patterns for fast queries)");
        }

        #[cfg(not(feature = "slm"))]
        {
            let _ = self.slm_query_classification_enabled; // Suppress unused warning
        }

        // Fallback to pattern-based classification
        eprintln!("[DEBUG-QP] Using pattern-based classification");
        Ok(self.intent_classifier.classify(query_text))
    }

    /// Execute a multi-dimensional query
    ///
    /// Classifies the query intent, retrieves results from relevant dimensions,
    /// and fuses them with adaptive weights.
    ///
    /// # Arguments
    ///
    /// * `query_text` - The natural language query
    /// * `query_embedding` - Vector embedding of the query
    /// * `limit` - Maximum number of results to return
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
    /// * `filters` - Optional metadata filters. All filters must match (AND logic)
    ///
    /// # Returns
    ///
    /// Tuple of (intent classification, fused results)
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        namespace: Option<&str>,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<(IntentClassification, Vec<FusedResult>, Vec<crate::query::profile_search::MatchedProfileFact>)> {
        // Step 1: Classify intent (try SLM first, fallback to patterns)
        let intent = self.classify_intent(query_text)?;

        // Step 1.5: Entity-first retrieval path (Sprint 18 Task 18.4)
        // Only fires for narrow entity_list_patterns (e.g., "What does Alice like?").
        // Broader entity queries go through the enhanced general path which preserves
        // multi-dimensional scoring while using profile sources and speaker reranking.
        if let Some(entity_name) = intent.entity_focus.clone() {
            if self.storage.find_entity_by_name(&entity_name)?.is_some() {
                return self.retrieve_entity_focused(
                    &entity_name,
                    query_text,
                    query_embedding,
                    limit,
                    namespace,
                    filters,
                    intent,
                );
            }
            // Entity not in graph — fall through to standard multi-dimensional retrieval
        }


        // Step 1.7: Detect entities for profile injection and speaker reranking.
        //
        // Two separate concerns that were previously coupled through a single target_entity:
        //   - query_entities (ALL detected): Used in Step 2.1 for profile source injection.
        //     Multi-entity queries ("How long have Mel and her husband been married?")
        //     need profile sources from ALL mentioned entities.
        //   - speaker_entity (single only): Used in Step 3.7 for speaker-aware reranking.
        //     Speaker penalty requires a single target — can't penalize speakers when the
        //     query is about multiple people.
        let query_entities = self.profile_search.detect_entities(query_text).unwrap_or_default();
        let speaker_entity = if query_entities.len() == 1 {
            Some(query_entities[0].clone())
        } else {
            None
        };

        // Step 2: Retrieve from all dimensions (fetch more to account for filtering)
        let needs_filtering =
            namespace.is_some() || (filters.is_some() && !filters.unwrap().is_empty());
        // When entity-specific retrieval is active, fetch more candidates to maintain
        // effective depth after speaker penalty or entity-based scoring.
        let fetch_multiplier = if needs_filtering || !query_entities.is_empty() { 5 } else { 3 };

        let mut semantic_scores =
            self.semantic_search(query_embedding, limit * fetch_multiplier)?;
        let mut bm25_scores = self.bm25_search(query_text, limit * fetch_multiplier)?;
        let mut temporal_scores = self.temporal_search(query_text, limit * fetch_multiplier)?;
        let mut causal_scores = self.causal_search(query_text, limit * fetch_multiplier)?;
        // Normalize entity_scores to partial MemoryId format immediately.
        // entity_search() returns full UUIDs (from storage), but semantic_search() returns
        // partial UUIDs (from vector index, first 8 bytes only). Without normalization the
        // same physical memory gets two entries in the fusion HashSet → duplicate results.
        // See: MemoryId::from_u64 / MemoryId::to_u64 for the partial-UUID convention.
        let mut entity_scores: HashMap<MemoryId, f32> = {
            let raw = self.entity_search(query_text, limit * fetch_multiplier)?;
            let mut norm: HashMap<MemoryId, f32> = HashMap::with_capacity(raw.len());
            for (id, score) in raw {
                norm.entry(MemoryId::from_u64(id.to_u64()))
                    .and_modify(|s: &mut f32| *s = (*s).max(score))
                    .or_insert(score);
            }
            norm
        };

        // Step 2.1: Add profile source memories to entity candidates with HIGH priority.
        // The entity graph only contains memories where entity names appear in text.
        // But when Melanie says "I like running", her name isn't in the text — only
        // the LLM extraction (with "Melanie says:" prefix) correctly attributes it.
        // Profile source_memories include these LLM-attributed memories.
        // Score 2.0 ensures they rank at the TOP of entity dimension after normalization,
        // giving them strong signal in RRF fusion. 2.0 baseline is REQUIRED — lower
        // values (tested 1.5) cause regression because wrong-entity graph memories
        // (0-1.0) compete too closely. Graduated scoring (2.0 + sem_bonus) also
        // regresses because it changes RRF normalization for all entity scores.
        //
        // Injects for ALL detected entities (not just single-entity queries).
        // Multi-entity queries need profile sources from every mentioned entity.
        for entity_name in &query_entities {
            if let Ok(Some(profile)) = self.storage.get_entity_profile(entity_name) {
                for source_id in &profile.source_memories {
                    // Normalize to partial ID so entity injection merges with semantic search.
                    // source_memories store full UUIDs; vector index uses partial (first 8 bytes).
                    entity_scores
                        .entry(MemoryId::from_u64(source_id.to_u64()))
                        .and_modify(|s| *s = (*s).max(2.0))
                        .or_insert(2.0);
                }
            }
        }

        // Step 2.2: Profile-based search for direct fact lookup (Phase 3)
        // Boost entity_scores with profile matches — memories that are sources of
        // matching facts get additional priority. Uses stemmed word-overlap so
        // "instruments" matches "instrument", "books" matches "book", etc.
        // Uses max() to ensure fact-matched memories rank AT LEAST as high as the
        // 2.0 baseline (never clamped below it).
        let profile_result = self.profile_search.search(query_text, query_embedding, limit * fetch_multiplier)?;
        let matched_facts = profile_result.matched_facts;
        for (memory_id, profile_score) in profile_result.source_scores {
            let boosted = 2.0 + profile_score; // fact-matched get 2.0-3.0
            // Normalize to partial ID (source_scores contain full UUIDs from storage).
            entity_scores
                .entry(MemoryId::from_u64(memory_id.to_u64()))
                .and_modify(|s| {
                    *s = (*s).max(boosted);
                })
                .or_insert(boosted);
        }

        // Normalize bm25/temporal/causal to partial MemoryId format before Step 2.3.
        // These searches return full UUIDs from storage; semantic uses partial UUIDs.
        // Normalization is required for Step 2.3's entity exemption check to work
        // correctly (partial semantic IDs must match entity_scores keys).
        let mut bm25_scores: HashMap<MemoryId, f32> = {
            let mut norm: HashMap<MemoryId, f32> = HashMap::with_capacity(bm25_scores.len());
            for (id, score) in bm25_scores {
                norm.entry(MemoryId::from_u64(id.to_u64()))
                    .and_modify(|s: &mut f32| *s = (*s).max(score))
                    .or_insert(score);
            }
            norm
        };
        let mut temporal_scores: HashMap<MemoryId, f32> = {
            let mut norm: HashMap<MemoryId, f32> = HashMap::with_capacity(temporal_scores.len());
            for (id, score) in temporal_scores {
                norm.entry(MemoryId::from_u64(id.to_u64()))
                    .and_modify(|s: &mut f32| *s = (*s).max(score))
                    .or_insert(score);
            }
            norm
        };
        let mut causal_scores: HashMap<MemoryId, f32> = {
            let mut norm: HashMap<MemoryId, f32> = HashMap::with_capacity(causal_scores.len());
            for (id, score) in causal_scores {
                norm.entry(MemoryId::from_u64(id.to_u64()))
                    .and_modify(|s: &mut f32| *s = (*s).max(score))
                    .or_insert(score);
            }
            norm
        };

        // Step 2.3: Adaptive pre-fusion semantic filtering
        // Aggregation queries ("What activities does X do?") need more recall — answer
        // docs mention specific instances with only moderate similarity to the category
        // query. Use 0.5x threshold for those. Extraction/Hypothetical keep strict
        // threshold for precision (avoids noise that hurts multi-hop reasoning).
        //
        // EXEMPTION: Memories with entity_score > 0 bypass this filter. These memories
        // matched a target entity (via profile source injection or entity graph), so they
        // are relevant even if semantically distant from the query phrasing. Without this,
        // ~60-80% of entity-matched candidates die before RRF fusion can help them.
        //
        // Preserve full semantic scores for accurate reporting in FusedResult.
        // The .retain() controls which memories contribute to the semantic RRF pathway,
        // but memories found via BM25/entity should still report their actual semantic score
        // (not 0.0) for diagnostics. After fusion, we patch scores from this full map.
        let semantic_scores_full = semantic_scores.clone();
        let effective_prefilter = if MultiTurnAggregator::is_aggregation(&query_text.to_lowercase()) {
            self.semantic_prefilter_threshold * 0.5
        } else {
            self.semantic_prefilter_threshold
        };
        if effective_prefilter > 0.0 {
            semantic_scores.retain(|id, score| {
                *score >= effective_prefilter
                    || entity_scores.get(id).copied().unwrap_or(0.0) > 0.0
            });
        }

        // Step 2.5: Filter by namespace if provided
        if let Some(ns) = namespace {
            self.filter_by_namespace(&mut semantic_scores, ns)?;
            self.filter_by_namespace(&mut bm25_scores, ns)?;
            self.filter_by_namespace(&mut temporal_scores, ns)?;
            self.filter_by_namespace(&mut causal_scores, ns)?;
            self.filter_by_namespace(&mut entity_scores, ns)?;
        }

        // Step 2.6: Filter by metadata if provided
        if let Some(filter_list) = filters {
            if !filter_list.is_empty() {
                self.filter_by_metadata(&mut semantic_scores, filter_list)?;
                self.filter_by_metadata(&mut bm25_scores, filter_list)?;
                self.filter_by_metadata(&mut temporal_scores, filter_list)?;
                self.filter_by_metadata(&mut causal_scores, filter_list)?;
                self.filter_by_metadata(&mut entity_scores, filter_list)?;
            }
        }

        // Step 2.7: Graph traversal expansion for retrieval augmentation (conditional)
        // Only expand if seed results have high quality scores (>0.6 threshold)
        // This prevents noise from over-expansion in session-based retrieval
        let mut seed_results: HashMap<MemoryId, f32> = semantic_scores.clone();
        for (id, score) in &bm25_scores {
            seed_results
                .entry(id.clone())
                .and_modify(|s: &mut f32| *s = (*s).max(*score))
                .or_insert(*score);
        }

        // Calculate average seed quality
        let avg_seed_quality = if !seed_results.is_empty() {
            seed_results.values().sum::<f32>() / seed_results.len() as f32
        } else {
            0.0
        };

        // Only perform graph traversal if seed quality is good (prevents noise)
        const GRAPH_EXPANSION_THRESHOLD: f32 = 0.6;
        if avg_seed_quality >= GRAPH_EXPANSION_THRESHOLD {
            let graph_traversal = crate::query::graph_traversal::GraphTraversal::new(
                self.graph_manager.clone(),
                self.storage.clone(),
                crate::query::graph_traversal::TraversalConfig::default(),
            );

            let expanded_results = graph_traversal.expand(&seed_results, intent.intent, limit * 5)?;

            // Merge expanded results into dimension scores
            // Expanded memories get added to entity/causal scores based on their source
            for (expanded_id, expansion_score) in expanded_results {
                // Normalize to partial ID before checking and inserting.
                // Graph traversal returns full UUIDs; score maps use partial UUIDs.
                let expanded_partial = MemoryId::from_u64(expanded_id.to_u64());
                // Skip if already in seed results (now correctly checks partial IDs)
                if seed_results.contains_key(&expanded_partial) {
                    continue;
                }

                // Add to appropriate dimension score maps based on intent
                match intent.intent {
                    crate::query::QueryIntent::Causal => {
                        causal_scores.insert(expanded_partial, expansion_score);
                    }
                    crate::query::QueryIntent::Entity | crate::query::QueryIntent::Factual => {
                        entity_scores.insert(expanded_partial, expansion_score);
                    }
                    crate::query::QueryIntent::Temporal => {
                        // Hybrid expansion - add to both temporal and entity
                        temporal_scores
                            .entry(expanded_partial.clone())
                            .and_modify(|s| *s = (*s).max(expansion_score * 0.7))
                            .or_insert(expansion_score * 0.7);
                        entity_scores.insert(expanded_partial, expansion_score * 0.5);
                    }
                }
            }
        }

        // Step 3: Fuse results with adaptive weights
        let mut fused_results = self.fusion_engine.fuse(
            intent.intent,
            &semantic_scores,
            &bm25_scores,
            &temporal_scores,
            &causal_scores,
            &entity_scores,
        );

        // Step 3.5: Cross-dimensional validation (Sprint 18 Task 18.2)
        // Calculate confidence based on how many dimensions contributed to each result
        // Multi-dimensional matches are more reliable than single-dimension matches
        //
        // Solution 4: Context-dependent confidence scoring
        // For Factual and Entity intents, single-dimension matches (often just semantic)
        // are expected and valid — don't penalize them as harshly as other intents.
        // This stops burying correct single-hop results below noisy multi-dimension matches.
        for result in &mut fused_results {
            // Count how many dimensions contributed (score > 0.0)
            let mut dimension_count = 0;

            if result.semantic_score > 0.0 {
                dimension_count += 1;
            }
            if result.bm25_score > 0.0 {
                dimension_count += 1;
            }
            if result.temporal_score > 0.0 {
                dimension_count += 1;
            }
            if result.causal_score > 0.0 {
                dimension_count += 1;
            }
            if result.entity_score > 0.0 {
                dimension_count += 1;
            }

            // Intent-dependent confidence: Factual/Entity queries often match on
            // only 1-2 dimensions (semantic for factual, entity for entity queries).
            // Using the default 0.4 penalty for these buries correct results.
            let single_dim_confidence = match intent.intent {
                crate::query::QueryIntent::Factual => 0.7,
                crate::query::QueryIntent::Entity => 0.7,
                _ => 0.4,
            };

            result.confidence = match dimension_count {
                5 => 1.0,   // All 5 dimensions agree - highest confidence
                4 => 0.9,   // 4 dimensions - very high confidence
                3 => 0.8,   // 3 dimensions - high confidence
                2 => 0.6,   // 2 dimensions - medium confidence
                1 => single_dim_confidence, // Intent-dependent (0.7 for factual/entity, 0.4 otherwise)
                _ => 0.2,   // 0 dimensions - very low confidence
            };

            // Adjust fused_score by confidence to penalize single-dimension matches
            result.fused_score *= result.confidence;
        }

        // Re-sort by adjusted fused_score (confidence-weighted)
        fused_results.sort_by(|a, b| {
            b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3.6: Patch semantic scores for accurate reporting
        // The pre-fusion .retain() removed low-scoring entries from semantic_scores,
        // so fusion reports 0.0 for memories found via BM25/entity. Restore the
        // actual semantic scores from the pre-filter snapshot for diagnostics.
        // This runs AFTER confidence scoring (Step 3.5) so it doesn't affect ranking —
        // patched scores are for reporting/diagnostics only.
        for result in &mut fused_results {
            if result.semantic_score == 0.0 {
                if let Some(&full_score) = semantic_scores_full.get(&result.id) {
                    result.semantic_score = full_score;
                }
            }
        }

        // Step 3.7: Speaker-aware reranking
        // When query asks about a specific person (e.g., "What did Melanie paint?"),
        // penalize results from other speakers. Semantic search matches on topic, not
        // information source — so "What's your best camping memory?" (asked by Caroline)
        // ranks high for "Where has Melanie camped?" despite containing no answer.
        // Penalizing non-target speakers corrects this without filtering.
        // Speaker reranking only fires for single-entity queries. Multi-entity queries
        // (e.g., "How long have Mel and her husband been married?") have no single target
        // speaker — applying a penalty would arbitrarily suppress one entity's memories.
        if let Some(ref speaker_target) = speaker_entity {
            let target_lower = speaker_target.to_lowercase();
            let mut any_speaker_data = false;
            for result in &mut fused_results {
                if let Ok(Some(memory)) = self.storage.get_memory_by_u64(result.id.to_u64()) {
                    if let Some(speaker) = memory.get_metadata("speaker") {
                        any_speaker_data = true;
                        if speaker.to_lowercase() != target_lower {
                            // Non-target speaker: strong penalty to suppress cross-entity
                            // pollution. In two-person conversations, the other person's
                            // memories dominate entity search because they mention the
                            // target's name. Tested: 0.2x gives best overall balance
                            // (single-hop +9pts, temporal preserved, R@20 +2.7pts).
                            result.fused_score *= 0.2;
                        }
                    }
                }
            }
            if any_speaker_data {
                fused_results.sort_by(|a, b| {
                    b.fused_score
                        .partial_cmp(&a.fused_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Step 3.8: Multi-evidence expansion for aggregation queries (Solution 5)
        // Aggregation queries ("What instruments does Melanie play?") need evidence
        // from 2-4 separate memories. Expand by fetching entity memories and adding
        // DIVERSE ones not already in results. Cap at 5 expansions for performance.
        if MultiTurnAggregator::is_aggregation(&query_text.to_lowercase()) {
            // Use speaker_entity for aggregation expansion (consistent with speaker reranking)
            if let Some(ref entity_name) = speaker_entity {
                if let Ok(Some(entity)) = self.storage.find_entity_by_name(entity_name) {
                    let graph = self.graph_manager.read().unwrap();
                    let entity_memories = graph.get_entity_memories(&entity.id);
                    drop(graph);

                    let existing_ids: std::collections::HashSet<_> =
                        fused_results.iter().map(|r| r.id.clone()).collect();

                    // Pre-load top-K embeddings once (avoid O(n*m) storage lookups)
                    let top_k_embeddings: Vec<Vec<f32>> = fused_results.iter()
                        .take(limit.min(10)) // Only compare against top-10
                        .filter_map(|r| {
                            self.storage.get_memory_by_u64(r.id.to_u64()).ok().flatten()
                                .map(|m| m.embedding.clone())
                        })
                        .filter(|e| !e.is_empty() && e.len() == query_embedding.len())
                        .collect();

                    let candidates: Vec<_> = entity_memories.memories.into_iter()
                        .filter(|m| !existing_ids.contains(m))
                        .collect();

                    let mut expanded = Vec::new();
                    let max_expansions = 5;

                    for candidate_id in candidates {
                        if expanded.len() >= max_expansions {
                            break;
                        }
                        if let Ok(Some(memory)) = self.storage.get_memory_by_u64(candidate_id.to_u64()) {
                            if memory.embedding.is_empty() || query_embedding.len() != memory.embedding.len() {
                                continue;
                            }

                            // Check diversity against pre-loaded embeddings
                            let max_sim = top_k_embeddings.iter()
                                .map(|e| cosine_similarity(&memory.embedding, e))
                                .fold(0.0f32, f32::max);

                            if max_sim < 0.8 {
                                let sem_score = cosine_similarity(query_embedding, &memory.embedding);
                                expanded.push(FusedResult {
                                    id: candidate_id,
                                    semantic_score: sem_score,
                                    bm25_score: 0.0,
                                    temporal_score: 0.0,
                                    causal_score: 0.0,
                                    entity_score: 0.5,
                                    fused_score: sem_score * 0.7,
                                    confidence: 0.7,
                                });
                            }
                        }
                    }

                    if !expanded.is_empty() {
                        fused_results.extend(expanded);
                        fused_results.sort_by(|a, b| {
                            b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
            }
        }

        // Step 3.9: MMR Diversity Reranking
        // After all scoring/penalties, apply Maximal Marginal Relevance to break ties
        // among entity-matched memories (168-205 memories at flat 2.0 compete randomly).
        // MMR ensures diverse context: pottery + camping + painting, not 5× pottery.
        // Formula: score = λ × rrf_score - (1-λ) × max_cosine(doc, already_selected)
        // λ=0.7 favors relevance, 0.3 weight on diversity.
        // Only applied when we have enough candidates to benefit from diversity.
        const MMR_LAMBDA: f32 = 0.7;
        const MMR_POOL_SIZE: usize = 50;
        if fused_results.len() > limit {
            let pool_size = fused_results.len().min(MMR_POOL_SIZE);
            let pool = &fused_results[..pool_size];

            // Load embeddings for the candidate pool
            let pool_embeddings: Vec<Option<Vec<f32>>> = pool
                .iter()
                .map(|r| {
                    self.storage
                        .get_memory_by_u64(r.id.to_u64())
                        .ok()
                        .flatten()
                        .map(|m| m.embedding)
                        .filter(|e| !e.is_empty() && e.len() == query_embedding.len())
                })
                .collect();

            let mut selected: Vec<usize> = Vec::with_capacity(limit);
            let mut remaining: Vec<usize> = (0..pool_size).collect();

            // Greedily select limit items by MMR score
            while selected.len() < limit && !remaining.is_empty() {
                let mut best_idx_in_remaining = 0;
                let mut best_mmr_score = f32::NEG_INFINITY;

                for (ri, &cand_idx) in remaining.iter().enumerate() {
                    let relevance = pool[cand_idx].fused_score;

                    // Max cosine similarity to any already-selected document
                    let max_sim = if selected.is_empty() {
                        0.0
                    } else if let Some(ref cand_emb) = pool_embeddings[cand_idx] {
                        selected
                            .iter()
                            .filter_map(|&sel_idx| {
                                pool_embeddings[sel_idx]
                                    .as_ref()
                                    .map(|sel_emb| cosine_similarity(cand_emb, sel_emb))
                            })
                            .fold(0.0f32, f32::max)
                    } else {
                        0.0 // No embedding — treat as fully diverse
                    };

                    let mmr_score = MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * max_sim;
                    if mmr_score > best_mmr_score {
                        best_mmr_score = mmr_score;
                        best_idx_in_remaining = ri;
                    }
                }

                let chosen = remaining.swap_remove(best_idx_in_remaining);
                selected.push(chosen);
            }

            // Rebuild fused_results in MMR order, then append any remaining beyond pool
            let mut mmr_results: Vec<FusedResult> =
                selected.into_iter().map(|i| pool[i].clone()).collect();
            if pool_size < fused_results.len() {
                mmr_results.extend(fused_results[pool_size..].iter().cloned());
            }
            fused_results = mmr_results;
        }

        // Step 4: Multi-turn aggregation for list/collection queries
        let aggregator = crate::query::aggregator::MultiTurnAggregator::default();
        let query_type = aggregator.classify_query(query_text);
        let mut final_results = aggregator.aggregate(
            query_type,
            query_text,
            fused_results,
            &self.storage,
            limit,
        )?;

        // Step 5: Adaptive-K (Top-p nucleus selection)
        if self.adaptive_k_threshold > 0.0 && self.adaptive_k_threshold < 1.0 && final_results.len() > 1 {
            let new_len = adaptive_k_select(&final_results, self.adaptive_k_threshold, limit);
            final_results.truncate(new_len);
        }

        Ok((intent, final_results, matched_facts))
    }

    /// Entity-first retrieval path (Sprint 18 Task 18.4)
    ///
    /// For entity-focused queries (e.g., "What does Alice like?"), this method:
    /// 1. Looks up the entity by name
    /// 2. Fetches ALL memories mentioning that entity (bypasses top-K limit)
    /// 3. Ranks by semantic similarity + keyword matching
    /// 4. Applies filters and returns top results
    ///
    /// This solves Category 1 failures where evidence is scattered across many turns.
    fn retrieve_entity_focused(
        &self,
        entity_name: &str,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        namespace: Option<&str>,
        filters: Option<&[MetadataFilter]>,
        intent: IntentClassification,
    ) -> Result<(IntentClassification, Vec<FusedResult>, Vec<crate::query::profile_search::MatchedProfileFact>)> {
        // Step 1: Find entity by name (case-insensitive)
        let entity = self.storage.find_entity_by_name(entity_name)?;

        if entity.is_none() {
            // Entity not found - return empty results
            return Ok((intent, vec![], vec![]));
        }

        let entity = entity.unwrap();

        // Step 2: Get ALL memories mentioning this entity from the graph
        let graph = self.graph_manager.read().unwrap();
        let entity_result = graph.get_entity_memories(&entity.id);
        let mut memory_id_set: std::collections::HashSet<MemoryId> =
            entity_result.memories.into_iter().collect();
        drop(graph); // Release lock

        // Step 2.1: Also include memories from entity PROFILE sources.
        // The entity graph only contains memories where the entity name appears in text.
        // But when Melanie says "I like running", her name isn't in the text — only the
        // LLM extraction (with "Melanie says:" prefix) knows it's about Melanie.
        // Profile source_memories include these LLM-attributed memories.
        if let Ok(Some(profile)) = self.storage.get_entity_profile(entity_name) {
            for source_id in &profile.source_memories {
                memory_id_set.insert(source_id.clone());
            }
        }

        if memory_id_set.is_empty() {
            return Ok((intent, vec![], vec![]));
        }

        let memory_ids: Vec<MemoryId> = memory_id_set.into_iter().collect();

        // Step 2.5: Look up entity profile for fact-based boosting (open-vocabulary)
        // Matches query words against ALL fact text (fact_type + value) for the entity.
        // No hardcoded keyword→fact_type mapping needed.
        let profile_boost_memories: HashMap<MemoryId, f32> =
            self.profile_search.compute_fact_boosts(entity_name, query_text, query_embedding)?;

        // Also get matched facts for synthetic injection
        let matched_facts = self.profile_search.search(query_text, query_embedding, limit)?.matched_facts;

        // Step 3: Calculate scores for each memory
        let mut scored_results = Vec::new();

        for memory_id in memory_ids {
            // Get memory from storage
            let memory = match self.storage.get_memory_by_u64(memory_id.to_u64())? {
                Some(m) => m,
                None => continue, // Skip if memory not found
            };

            // Apply namespace filter
            if let Some(ns) = namespace {
                if memory.get_namespace() != ns {
                    continue;
                }
            }

            // Apply metadata filters
            if let Some(filter_list) = filters {
                if !filter_list.is_empty() {
                    if !Self::memory_matches_filters(&memory, filter_list) {
                        continue;
                    }
                }
            }

            // Calculate semantic similarity
            let semantic_score = if !memory.embedding.is_empty() && query_embedding.len() == memory.embedding.len() {
                cosine_similarity(query_embedding, &memory.embedding)
            } else {
                0.0
            };

            // Calculate BM25 score for keyword matching
            let bm25_score = {
                let results = self.bm25_index.search(query_text, 1000)?;
                results.iter()
                    .find(|r| r.memory_id == memory_id)
                    .map(|r| r.score)
                    .unwrap_or(0.0)
            };

            // Check for profile-based boost
            let profile_boost = profile_boost_memories
                .get(&memory_id)
                .copied()
                .unwrap_or(0.0);

            // Speaker-aware scoring: prioritize memories where the target entity is speaking.
            // In a two-person conversation, the entity graph includes memories where the OTHER
            // person mentions the target entity. Those memories contain the other person's info,
            // not the target's. Penalize non-target speakers to fix this inversion.
            let speaker_factor = match memory.get_metadata("speaker") {
                Some(speaker) if speaker.to_lowercase() == entity_name.to_lowercase() => 1.0,
                Some(_) => 0.4, // Non-target speaker: reduce score
                None => 0.7,    // No speaker metadata: neutral
            };

            // Combined score: semantic + BM25 + profile boost, adjusted by speaker
            let base_score = 0.7 * semantic_score + 0.3 * bm25_score;
            let combined_score = if profile_boost > 0.0 {
                // Significant boost for profile matches - these are direct fact sources
                (base_score + profile_boost * 0.5).min(1.0) * speaker_factor
            } else {
                base_score * speaker_factor
            };

            // Higher confidence for profile-matched memories
            let confidence = if profile_boost > 0.0 { 0.95 } else { 0.9 };

            scored_results.push(FusedResult {
                id: memory_id.clone(),
                semantic_score,
                bm25_score,
                temporal_score: 0.0,
                causal_score: 0.0,
                entity_score: 1.0 + profile_boost, // Boost entity score for profile matches
                fused_score: combined_score,
                confidence,
            });
        }

        // Step 4: Sort by combined score and take top-K
        scored_results.sort_by(|a, b| {
            b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        let final_results = scored_results.into_iter().take(limit).collect();

        Ok((intent, final_results, matched_facts))
    }

    /// Extract the primary person entity from query text.
    ///
    /// Uses profile-aware entity detection (case-insensitive whole-word matching
    /// against stored entity profile names). Returns Some(name) only when exactly
    /// one entity is found — queries mentioning multiple people return None.
    ///
    /// Note: No longer used in the main query path (Step 2.1 now injects for ALL
    /// detected entities via query_entities). Kept for tests and potential callers.
    #[allow(dead_code)]
    fn extract_query_entity(&self, query_text: &str) -> Option<String> {
        let entities = self.profile_search.detect_entities(query_text).ok()?;
        if entities.len() == 1 {
            Some(entities[0].clone())
        } else {
            None
        }
    }

    /// Perform semantic similarity search
    fn semantic_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<HashMap<MemoryId, f32>> {
        let index = self.vector_index.read().unwrap();
        let results = index.search(query_embedding, limit)?;

        let mut scores = HashMap::new();
        for result in results {
            scores.insert(result.id, result.similarity);
        }

        Ok(scores)
    }

    /// Perform BM25 keyword search
    ///
    /// Searches using BM25 algorithm for exact term matching.
    /// Returns empty if query has no valid terms (too short or all stopwords).
    fn bm25_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let results = self.bm25_index.search(query_text, limit)?;

        let mut scores = HashMap::new();
        for result in results {
            scores.insert(result.memory_id, result.score);
        }

        // Normalize scores to 0.0-1.0 range
        FusionEngine::normalize_scores(&mut scores);

        Ok(scores)
    }

    /// Perform temporal search based on SLM metadata with pattern fallback
    ///
    /// Uses SLM-extracted temporal metadata (markers, sequence, relative_time, absolute_dates)
    /// when available, falling back to pattern-based temporal content scoring.
    /// Falls back to weak recency signal if no temporal context found.
    fn temporal_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        // Get more memories to search for temporal matches
        // Use larger multiplier to search broader corpus for better recall
        let fetch_limit = limit * 20;
        let recent_memories = self.temporal_index.recent(fetch_limit)?;

        let mut scores = HashMap::new();

        for temporal_result in recent_memories {
            // Load memory to access metadata
            if let Some(memory) = self.storage.get_memory(&temporal_result.id)? {
                let mut temporal_score = 0.0;

                // Try SLM metadata first (richer temporal information)
                if let Some(slm_metadata) = Self::get_slm_metadata(&memory) {
                    temporal_score = Self::calculate_slm_temporal_score(query_text, &slm_metadata);
                }

                // Fallback to pattern-based temporal matching if SLM didn't match
                if temporal_score == 0.0 {
                    // Use the temporal index's content-based scoring
                    if let Some(content_score) = self.get_temporal_content_score(&memory, query_text) {
                        temporal_score = content_score;
                    }
                }

                if temporal_score > 0.0 {
                    scores.insert(temporal_result.id.clone(), temporal_score);
                }
            }
        }

        // If we found temporal matches, return them
        if !scores.is_empty() {
            // Normalize scores to 0.0-1.0 range
            FusionEngine::normalize_scores(&mut scores);

            // Take top results by score
            let mut score_vec: Vec<_> = scores.into_iter().collect();
            score_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            score_vec.truncate(limit);

            Ok(score_vec.into_iter().collect())
        } else {
            // No temporal matches found - fall back to weak recency signal
            self.temporal_search_recency_fallback(limit)
        }
    }

    /// Get temporal content score for a memory using pattern-based matching
    ///
    /// Checks if the memory's temporal expressions match the query's temporal expressions.
    fn get_temporal_content_score(&self, memory: &Memory, query_text: &str) -> Option<f32> {
        // Get memory's temporal expressions from metadata
        let temporal_expressions_json = memory.get_metadata("temporal_expressions")?;
        let memory_expressions: Vec<String> = serde_json::from_str(temporal_expressions_json).ok()?;

        if memory_expressions.is_empty() {
            return None;
        }

        // Extract temporal expressions from query using the same patterns
        let query_lower = query_text.to_lowercase();

        // Common temporal keywords to match
        let temporal_keywords = [
            "yesterday", "today", "tomorrow", "last week", "next week",
            "last month", "next month", "last year", "next year",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "morning", "afternoon", "evening", "night",
            "recently", "earlier", "later", "before", "after",
            "first", "last", "initially", "finally", "eventually",
        ];

        let mut match_score: f32 = 0.0;

        for memory_expr in &memory_expressions {
            let expr_lower = memory_expr.to_lowercase();

            // Check direct match
            if query_lower.contains(&expr_lower) {
                match_score += 1.0;
            } else {
                // Check for keyword overlap
                for keyword in &temporal_keywords {
                    if query_lower.contains(keyword) && expr_lower.contains(keyword) {
                        match_score += 0.7;
                        break;
                    }
                }
            }
        }

        if match_score > 0.0 {
            Some((match_score / memory_expressions.len() as f32).min(1.0))
        } else {
            None
        }
    }

    /// Fallback temporal search using hybrid half-life + rank tiebreaker.
    ///
    /// Used when query has no temporal context or no temporal matches found.
    /// Combines two signals:
    /// - **Half-life decay** (96% weight): `0.5^(age_days / 365)` — real temporal
    ///   distance. Memories from yesterday score much higher than memories from
    ///   last year. Essential for production systems spanning months/years.
    /// - **Rank tiebreaker** (4% weight): linear rank-based nudge that breaks ties
    ///   when timestamps are clustered (e.g., batch-ingested DBs where all memories
    ///   have similar timestamps).
    ///
    /// Scores are scaled to a weak 0.0-0.3 range with a 5% floor so memories
    /// never fully vanish. This keeps temporal as a gentle signal that doesn't
    /// dominate other dimensions (semantic, entity, BM25).
    fn temporal_search_recency_fallback(&self, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let results = self.temporal_index.recent(limit)?;

        let count = results.len();
        let mut scores = HashMap::new();
        let now_micros = crate::types::Timestamp::now().as_micros();

        const HALF_LIFE_DAYS: f64 = 365.0;
        const MICROS_PER_DAY: f64 = 86_400.0 * 1_000_000.0;
        const FLOOR: f64 = 0.05;

        for (i, result) in results.into_iter().enumerate() {
            let score = if count <= 1 {
                0.3
            } else {
                // Use event_date from metadata when available, fall back to created_at.
                // event_date gives the actual date the event occurred (extracted by LLM),
                // while created_at is just when the memory was added to the DB.
                // For batch-ingested DBs, all created_at values are nearly identical,
                // making the temporal dimension useless without event_date.
                let effective_micros = self
                    .storage
                    .get_memory(&result.id)
                    .ok()
                    .flatten()
                    .and_then(|m| m.get_metadata("event_date").cloned())
                    .and_then(|d| Timestamp::from_iso8601_date(&d))
                    .map(|ts| ts.as_micros())
                    .unwrap_or(result.timestamp.as_micros());

                let age_micros = now_micros.saturating_sub(effective_micros);
                let age_days = age_micros as f64 / MICROS_PER_DAY;
                let half_life = 0.5_f64.powf(age_days / HALF_LIFE_DAYS);

                // Rank-based tiebreaker (newest=1.0, oldest=0.0)
                let rank_signal = 1.0 - (i as f64 / (count - 1) as f64);

                // Blend: 96% half-life + 4% rank tiebreaker
                let blended = 0.96 * half_life + 0.04 * rank_signal;

                // Scale to 0.3 range with floor
                (0.3 * (FLOOR + (1.0 - FLOOR) * blended)) as f32
            };
            if score > 0.01 {
                scores.insert(result.id, score);
            }
        }

        Ok(scores)
    }

    /// Perform entity-based search using entity graph with BM25 fallback
    ///
    /// Solution 1: Uses the entity graph (built at ingestion time) to find ALL
    /// memories mentioning each query entity across the entire corpus, not just
    /// recent ones. Graph provides CANDIDATES; scoring uses metadata overlap.
    ///
    /// Solution 2: When an entity is not in the graph (extraction missed it),
    /// falls back to BM25 keyword search for the entity name.
    fn entity_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        // Extract entities from query using profile-aware detection
        let query_entities = self.profile_search.detect_entities(query_text)?;

        // If query has no entities, return empty (no entity scoring)
        if query_entities.is_empty() {
            return Ok(HashMap::new());
        }

        // Convert query entities to lowercase set for case-insensitive matching
        let query_entity_set: std::collections::HashSet<String> =
            query_entities.iter().map(|e| e.to_lowercase()).collect();

        let mut scores = HashMap::new();
        let mut found_via_graph = false;

        // Solution 1: Full-corpus entity search via graph lookup
        // Use graph to find CANDIDATES across the entire corpus,
        // then score each using metadata overlap (preserves nuanced ranking).
        {
            let graph = self.graph_manager.read().unwrap();
            let mut candidate_ids = Vec::new();
            let mut relationship_candidates: std::collections::HashSet<MemoryId> =
                std::collections::HashSet::new();

            for query_entity in &query_entities {
                if let Ok(Some(entity)) = self.storage.find_entity_by_name(query_entity) {
                    found_via_graph = true;
                    let entity_result = graph.get_entity_memories(&entity.id);
                    for memory_id in entity_result.memories {
                        candidate_ids.push(memory_id);
                    }

                    // Relationship traversal: follow Entity→Entity edges
                    let related = graph.get_related_entities(&entity.id);
                    for (related_entity_id, _relation_type) in related {
                        let related_memories = graph.get_entity_memories(&related_entity_id);
                        for memory_id in related_memories.memories {
                            relationship_candidates.insert(memory_id.clone());
                            candidate_ids.push(memory_id);
                        }
                    }
                }
            }

            drop(graph); // Release lock before storage lookups

            // Score each candidate using metadata overlap (same quality as original)
            for memory_id in candidate_ids {
                if let Ok(Some(memory)) = self.storage.get_memory(&memory_id) {
                    let mut entity_score = 0.0;

                    // Try SLM metadata first (richer entity information)
                    if let Some(slm_metadata) = Self::get_slm_metadata(&memory) {
                        entity_score = Self::calculate_slm_entity_score(&query_entity_set, &slm_metadata);
                    }

                    // Fallback to entity_names metadata
                    if entity_score == 0.0 {
                        if let Some(entities_json) = memory.get_metadata("entity_names") {
                            if let Ok(memory_entities) = serde_json::from_str::<Vec<String>>(entities_json) {
                                entity_score = Self::calculate_entity_overlap(&query_entity_set, &memory_entities);
                            }
                        }
                    }

                    // Floor score: 0.5 for direct graph-confirmed, 0.4 for relationship-traversed
                    if entity_score == 0.0 {
                        entity_score = if relationship_candidates.contains(&memory_id) {
                            0.4
                        } else {
                            0.5
                        };
                    }

                    scores
                        .entry(memory_id)
                        .and_modify(|s: &mut f32| *s = (*s).max(entity_score))
                        .or_insert(entity_score);
                }
            }
        }

        // Solution 2: BM25 keyword fallback for entity search
        // If entity not found in graph, search for entity name as a keyword.
        if !found_via_graph {
            for query_entity in &query_entities {
                let bm25_results = self.bm25_index.search(query_entity, limit * 5)?;
                for result in bm25_results {
                    if result.score > 0.0 {
                        let normalized = result.score / (result.score + 1.0);
                        scores
                            .entry(result.memory_id)
                            .and_modify(|s: &mut f32| *s = (*s + normalized).min(1.0))
                            .or_insert(normalized);
                    }
                }
            }
        }

        // Normalize and return top results
        if !scores.is_empty() {
            FusionEngine::normalize_scores(&mut scores);

            let mut score_vec: Vec<_> = scores.into_iter().collect();
            score_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            score_vec.truncate(limit);

            Ok(score_vec.into_iter().collect())
        } else {
            Ok(HashMap::new())
        }
    }

    /// Calculate entity overlap score between query and memory entities
    ///
    /// Returns score from 0.0 to 1.0 based on:
    /// - Exact matches (case-insensitive): 1.0 per match
    /// - Partial matches (substring): 0.5 per match
    /// - Average across query entities
    fn calculate_entity_overlap(
        query_entity_set: &std::collections::HashSet<String>,
        memory_entities: &[String],
    ) -> f32 {
        if query_entity_set.is_empty() || memory_entities.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;

        // Convert memory entities to lowercase for comparison
        let memory_entity_set: std::collections::HashSet<String> =
            memory_entities.iter().map(|e| e.to_lowercase()).collect();

        for query_entity in query_entity_set {
            // Check for exact match (case-insensitive)
            if memory_entity_set.contains(query_entity) {
                total_score += 1.0;
            } else {
                // Check for partial match (substring)
                for memory_entity in &memory_entity_set {
                    if memory_entity.contains(query_entity) || query_entity.contains(memory_entity)
                    {
                        total_score += 0.5;
                        break;
                    }
                }
            }
        }

        // Average score across query entities
        if query_entity_set.len() > 0 {
            total_score / query_entity_set.len() as f32
        } else {
            0.0
        }
    }

    /// Perform profile-based search using Entity Profiles
    ///
    /// Looks up Entity Profiles for entities mentioned in the query and
    /// returns source memories of matching facts. This enables direct fact
    /// Perform causal language-based search using SLM metadata with pattern fallback
    ///
    /// Uses SLM-extracted causal metadata (relationships, density, implicit causation)
    /// when available, falling back to pattern-based causal density.
    /// Returns empty if query has no causal intent.
    fn causal_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let causal_extractor = get_causal_extractor();

        // Check if query has causal intent
        if !causal_extractor.has_causal_intent(query_text) {
            // No causal focus → no causal scoring
            return Ok(HashMap::new());
        }

        // Get more memories to search for causal matches
        // Use larger multiplier to search broader corpus for better recall
        let fetch_limit = limit * 20;
        let recent_memories = self.temporal_index.recent(fetch_limit)?;

        let mut scores = HashMap::new();

        for temporal_result in recent_memories {
            // Load memory to access metadata
            if let Some(memory) = self.storage.get_memory(&temporal_result.id)? {
                let mut causal_score = 0.0;

                // Try SLM metadata first (richer causal information)
                if let Some(slm_metadata) = Self::get_slm_metadata(&memory) {
                    causal_score = Self::calculate_slm_causal_score(query_text, &slm_metadata);
                }

                // Fallback to pattern-based causal_density if SLM didn't match
                if causal_score == 0.0 {
                    if let Some(density_str) = memory.get_metadata("causal_density") {
                        if let Ok(causal_density) = density_str.parse::<f32>() {
                            // Only score memories with significant causal density (> 0.1)
                            if causal_density > 0.1 {
                                // Check if memory has causal graph links for boost
                                let has_graph_links = {
                                    let graph = self.graph_manager.read().unwrap();
                                    graph.get_causes(&temporal_result.id, 1).ok().map_or(false, |r| !r.paths.is_empty())
                                        || graph.get_effects(&temporal_result.id, 1).ok().map_or(false, |r| !r.paths.is_empty())
                                };

                                // Calculate relevance score with optional graph boost
                                causal_score =
                                    causal_extractor.calculate_relevance_score(causal_density, has_graph_links);
                            }
                        }
                    }
                }

                if causal_score > 0.0 {
                    scores.insert(temporal_result.id.clone(), causal_score);
                }
            }
        }

        // If we found causal matches, return them
        if !scores.is_empty() {
            // Normalize scores to 0.0-1.0 range
            FusionEngine::normalize_scores(&mut scores);

            // Take top results by score
            let mut score_vec: Vec<_> = scores.into_iter().collect();
            score_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            score_vec.truncate(limit);

            Ok(score_vec.into_iter().collect())
        } else {
            // No causal matches found
            Ok(HashMap::new())
        }
    }

    /// Perform temporal range search
    ///
    /// Search memories within a specific time range
    ///
    /// # Arguments
    ///
    /// * `start` - Start of time range
    /// * `end` - End of time range
    /// * `limit` - Maximum number of results
    /// * `namespace` - Optional namespace filter
    pub fn temporal_range_query(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<FusedResult>> {
        // Fetch more results if filtering by namespace
        let fetch_limit = if namespace.is_some() {
            limit * 3
        } else {
            limit
        };
        let results = self.temporal_index.range_query(start, end, fetch_limit)?;

        // Convert to fused results with temporal scores
        let mut fused: Vec<FusedResult> = Vec::new();
        let mut position = 0;

        for result in results {
            // Filter by namespace if provided
            if let Some(ns) = namespace {
                if let Ok(Some(memory)) = self.storage.get_memory(&result.id) {
                    if memory.get_namespace() != ns {
                        continue; // Skip memories not in this namespace
                    }
                } else {
                    continue; // Skip if memory not found
                }
            }

            // Calculate temporal score based on position
            let temporal_score = if fetch_limit > 1 {
                1.0 - (position as f32 / (fetch_limit - 1) as f32)
            } else {
                1.0
            };

            fused.push(FusedResult {
                id: result.id,
                semantic_score: 0.0,
                bm25_score: 0.0,
                temporal_score,
                causal_score: 0.0,
                entity_score: 0.0,
                fused_score: temporal_score,
                confidence: 1.0,
            });

            position += 1;
            if fused.len() >= limit {
                break;
            }
        }

        Ok(fused)
    }

    /// Filter a score map by namespace
    ///
    /// Removes entries whose memories are not in the specified namespace
    fn filter_by_namespace(
        &self,
        scores: &mut HashMap<MemoryId, f32>,
        namespace: &str,
    ) -> Result<()> {
        // Collect IDs to remove (can't remove during iteration)
        let mut to_remove = Vec::new();

        for memory_id in scores.keys() {
            // Use get_memory_by_u64 to support both full and partial (normalized) MemoryIds.
            if let Some(memory) = self.storage.get_memory_by_u64(memory_id.to_u64())? {
                if memory.get_namespace() != namespace {
                    to_remove.push(memory_id.clone());
                }
            } else {
                // Memory doesn't exist, remove it
                to_remove.push(memory_id.clone());
            }
        }

        // Remove filtered-out IDs
        for id in to_remove {
            scores.remove(&id);
        }

        Ok(())
    }

    /// Filter a score map by metadata filters
    ///
    /// Removes entries whose memories don't match ALL provided filters
    fn filter_by_metadata(
        &self,
        scores: &mut HashMap<MemoryId, f32>,
        filters: &[MetadataFilter],
    ) -> Result<()> {
        if filters.is_empty() {
            return Ok(());
        }

        // Collect IDs to remove (can't remove during iteration)
        let mut to_remove = Vec::new();

        for memory_id in scores.keys() {
            // Use get_memory_by_u64 to support both full and partial (normalized) MemoryIds.
            if let Some(memory) = self.storage.get_memory_by_u64(memory_id.to_u64())? {
                // Check if memory matches ALL filters
                if !Self::memory_matches_filters(&memory, filters) {
                    to_remove.push(memory_id.clone());
                }
            } else {
                // Memory doesn't exist, remove it
                to_remove.push(memory_id.clone());
            }
        }

        // Remove filtered-out IDs
        for id in to_remove {
            scores.remove(&id);
        }

        Ok(())
    }

    /// Check if a memory matches all metadata filters
    fn memory_matches_filters(memory: &Memory, filters: &[MetadataFilter]) -> bool {
        for filter in filters {
            let value = memory.metadata.get(&filter.field).map(|s| s.as_str());
            if !filter.matches(value) {
                return false; // Any filter fails = memory doesn't match
            }
        }
        true // All filters passed
    }

    /// Parse SlmMetadata from memory's slm_metadata field
    ///
    /// Returns None if the memory doesn't have slm_metadata or if parsing fails.
    /// This allows graceful fallback to pattern-based matching.
    fn get_slm_metadata(memory: &Memory) -> Option<SlmMetadata> {
        memory
            .get_metadata("slm_metadata")
            .and_then(|json| serde_json::from_str(json).ok())
    }

    /// Calculate entity match score using SLM metadata
    ///
    /// Matches query entities against SLM-extracted entities, considering:
    /// - Entity names (exact and partial match)
    /// - Entity mentions (pronoun resolution)
    /// - Entity roles and types
    fn calculate_slm_entity_score(
        query_entities: &std::collections::HashSet<String>,
        slm_metadata: &SlmMetadata,
    ) -> f32 {
        if query_entities.is_empty() || slm_metadata.entities.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut matches = 0;

        for query_entity in query_entities {
            let query_lower = query_entity.to_lowercase();

            for extracted in &slm_metadata.entities {
                // Check exact name match
                if extracted.name.to_lowercase() == query_lower {
                    total_score += 1.0;
                    matches += 1;
                    break;
                }

                // Check mentions (e.g., "she", "he", "they" resolved to entity)
                for mention in &extracted.mentions {
                    if mention.to_lowercase() == query_lower {
                        total_score += 0.9; // Slightly lower for mention match
                        matches += 1;
                        break;
                    }
                }

                // Check partial match (substring)
                if extracted.name.to_lowercase().contains(&query_lower)
                    || query_lower.contains(&extracted.name.to_lowercase())
                {
                    total_score += 0.5;
                    matches += 1;
                    break;
                }
            }
        }

        if matches > 0 {
            total_score / query_entities.len() as f32
        } else {
            0.0
        }
    }

    /// Calculate temporal match score using SLM metadata
    ///
    /// Matches query temporal expressions against SLM-extracted temporal metadata:
    /// - Temporal markers (yesterday, last week, etc.)
    /// - Sequence position (early, middle, late)
    /// - Relative time (before current, concurrent, after current)
    /// - Absolute dates
    fn calculate_slm_temporal_score(
        query_text: &str,
        slm_metadata: &SlmMetadata,
    ) -> f32 {
        let query_lower = query_text.to_lowercase();
        let temporal = &slm_metadata.temporal;
        let mut score: f32 = 0.0;

        // Check for temporal marker matches
        for marker in &temporal.markers {
            if query_lower.contains(&marker.to_lowercase()) {
                score += 0.8;
                break; // One match is enough
            }
        }

        // Check for absolute date matches
        for date in &temporal.absolute_dates {
            if query_lower.contains(&date.to_lowercase()) {
                score += 0.9;
                break;
            }
        }

        // Check for sequence-related queries
        if let Some(ref sequence) = temporal.sequence {
            let seq_lower = sequence.to_lowercase();
            // Match queries like "first", "initially", "beginning" to "early"
            if (query_lower.contains("first") || query_lower.contains("initial") || query_lower.contains("begin"))
                && seq_lower == "early"
            {
                score += 0.7;
            }
            // Match queries like "last", "finally", "end" to "late"
            if (query_lower.contains("last") || query_lower.contains("final") || query_lower.contains("end"))
                && seq_lower == "late"
            {
                score += 0.7;
            }
        }

        // Check for relative time matches
        if let Some(ref relative) = temporal.relative_time {
            let rel_lower = relative.to_lowercase();
            if query_lower.contains("before") && rel_lower.contains("before") {
                score += 0.6;
            }
            if query_lower.contains("after") && rel_lower.contains("after") {
                score += 0.6;
            }
            if (query_lower.contains("recent") || query_lower.contains("latest"))
                && rel_lower.contains("current")
            {
                score += 0.5;
            }
        }

        score.min(1.0) // Cap at 1.0
    }

    /// Calculate causal match score using SLM metadata
    ///
    /// Matches based on:
    /// - Causal relationships (cause/effect pairs)
    /// - Causal density
    /// - Implicit causation detection
    fn calculate_slm_causal_score(
        query_text: &str,
        slm_metadata: &SlmMetadata,
    ) -> f32 {
        let causal = &slm_metadata.causal;
        let query_lower = query_text.to_lowercase();

        // Base score from causal density
        let mut score = causal.density * 0.5;

        // Boost for explicit causal markers
        if !causal.explicit_markers.is_empty() {
            score += 0.2;
        }

        // Boost for implicit causation (captures nuanced causal relationships)
        if causal.has_implicit_causation {
            score += 0.3;
        }

        // Check if query terms match any cause/effect in relationships
        for rel in &causal.relationships {
            let cause_lower = rel.cause.to_lowercase();
            let effect_lower = rel.effect.to_lowercase();

            // Check if query mentions the cause or effect
            if query_lower.contains(&cause_lower) || cause_lower.contains(&query_lower) {
                score += 0.3 * rel.confidence;
            }
            if query_lower.contains(&effect_lower) || effect_lower.contains(&query_lower) {
                score += 0.3 * rel.confidence;
            }
        }

        score.min(1.0) // Cap at 1.0
    }
}

/// Calculate cosine similarity between two vectors
///
/// Returns similarity in range [0.0, 1.0] where 1.0 is identical vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    (dot_product / (magnitude_a * magnitude_b)).max(0.0).min(1.0)
}

/// Adaptive-K (Top-p nucleus selection) for dynamic result count.
///
/// Instead of always returning `limit` results, dynamically selects based on
/// fused score distribution. Converts fused_scores to probabilities via softmax,
/// then returns results until cumulative probability >= threshold.
///
/// This prevents low-quality padding from diluting the context window.
/// Research basis: Calvin Ku's Adaptive-K fork of EmergenceMem Simple Fast.
///
/// Returns the number of results to keep (bounded by [limit/3, results.len()]).
fn adaptive_k_select(results: &[FusedResult], threshold: f32, limit: usize) -> usize {
    if results.len() <= 1 {
        return results.len();
    }

    let min_k = (limit / 3).max(5).min(results.len());

    // Softmax with score shifting for numerical stability
    let max_score = results.iter()
        .map(|r| r.fused_score)
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_scores: Vec<f32> = results.iter()
        .map(|r| (r.fused_score - max_score).exp())
        .collect();

    let sum_exp: f32 = exp_scores.iter().sum();

    if sum_exp <= 0.0 {
        return results.len();
    }

    let mut cumulative = 0.0f32;
    for (i, &exp_s) in exp_scores.iter().enumerate() {
        cumulative += exp_s / sum_exp;
        if cumulative >= threshold {
            return (i + 1).max(min_k);
        }
    }

    results.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::GraphManager,
        index::{VectorIndex, VectorIndexConfig},
        storage::StorageEngine,
        types::Memory,
    };
    use tempfile::tempdir;

    fn create_test_planner() -> (QueryPlanner, tempfile::TempDir) {
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

        // Use 0.0 thresholds for tests to avoid filtering test results
        // Use Weighted strategy for tests to maintain existing test expectations
        let planner = QueryPlanner::new(
            storage,
            vector_index,
            bm25_index,
            temporal_index,
            graph_manager,
            0.0, // fusion_semantic_threshold
            0.0, // semantic_prefilter_threshold
            crate::query::FusionStrategy::Weighted, // strategy (weighted for backward compat in tests)
            60.0, // rrf_k (not used with Weighted strategy)
            #[cfg(feature = "slm")]
            None, // slm_config (disabled for tests)
            false, // slm_query_classification_enabled (disabled for fast tests)
            0.0,  // adaptive_k_threshold (disabled for tests)
        ).expect("Failed to create QueryPlanner");

        (planner, dir)
    }

    #[test]
    fn test_query_planner_creation() {
        let (_planner, _dir) = create_test_planner();
        // Just verify it creates without panic
    }

    #[test]
    fn test_semantic_search() {
        let (planner, _dir) = create_test_planner();

        // Add a memory
        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let mem_id = memory.id.clone();
        planner.storage.store_memory(&memory).unwrap();
        {
            let mut index = planner.vector_index.write().unwrap();
            index.add(mem_id.clone(), &memory.embedding).unwrap();
        }

        // Search
        let scores = planner.semantic_search(&vec![0.1; 384], 10).unwrap();
        assert!(!scores.is_empty());
        // The scores map should contain at least one result
        assert!(scores.len() > 0);
    }

    #[test]
    fn test_temporal_search() {
        let (planner, _dir) = create_test_planner();

        // Add memories
        let mem1 = Memory::new("Memory 1".to_string(), vec![0.1; 384]);
        let mem2 = Memory::new("Memory 2".to_string(), vec![0.2; 384]);

        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();

        // Also add to temporal index (normally done by ingestion pipeline)
        planner
            .temporal_index
            .add(&mem1.id, mem1.created_at)
            .unwrap();
        planner
            .temporal_index
            .add(&mem2.id, mem2.created_at)
            .unwrap();

        // Search with query text (no temporal expressions = fallback to recency)
        let scores = planner.temporal_search("test query", 10).unwrap();
        assert_eq!(scores.len(), 2);

        // With fallback, most recent should have higher score (but weak signal)
        let score1 = scores[&mem1.id];
        let score2 = scores[&mem2.id];
        assert!(score2 >= score1); // mem2 was added later
    }

    #[test]
    fn test_entity_search() {
        let (planner, _dir) = create_test_planner();

        // This test just verifies entity_search doesn't crash with no entities
        let scores = planner.entity_search("Show me Alice", 10).unwrap();
        // No entities exist, so should be empty
        assert_eq!(scores.len(), 0);
    }

    #[test]
    fn test_full_query() {
        let (planner, _dir) = create_test_planner();

        // Add a memory
        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let mem_id = memory.id.clone();
        planner.storage.store_memory(&memory).unwrap();
        {
            let mut index = planner.vector_index.write().unwrap();
            index.add(mem_id.clone(), &memory.embedding).unwrap();
        }
        // Add to temporal index too
        planner
            .temporal_index
            .add(&mem_id, memory.created_at)
            .unwrap();

        // Execute query
        let (intent, results, _matched_facts) = planner
            .query("test query", &vec![0.1; 384], 10, None, None)
            .unwrap();

        // Should classify as factual
        assert_eq!(intent.intent, crate::query::intent::QueryIntent::Factual);

        // Should have results
        assert!(!results.is_empty());
        // Results should contain our memory (may not be first if other memories exist)
        // Compare by u64 key: partial (normalized) and full UUIDs share the same first 8 bytes.
        assert!(results.iter().any(|r| r.id.to_u64() == mem_id.to_u64()));
    }

    #[test]
    fn test_temporal_range_query() {
        let (planner, _dir) = create_test_planner();

        let now = Timestamp::now();
        let mem1 = Memory::new_with_timestamp(
            "Memory 1".to_string(),
            vec![0.1; 384],
            now.subtract_days(1),
        );

        planner.storage.store_memory(&mem1).unwrap();
        // Add to temporal index
        planner
            .temporal_index
            .add(&mem1.id, mem1.created_at)
            .unwrap();

        let results = planner
            .temporal_range_query(now.subtract_days(2), now, 10, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, mem1.id);
    }

    #[test]
    fn test_query_with_namespace_filtering() {
        let (planner, _dir) = create_test_planner();

        // Create memories in different namespaces
        let mut mem1 = Memory::new("NS1 memory".to_string(), vec![0.1; 384]);
        mem1.set_namespace("ns1");
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("NS2 memory".to_string(), vec![0.15; 384]);
        mem2.set_namespace("ns2");
        let mem2_id = mem2.id.clone();

        let mem3 = Memory::new("Default memory".to_string(), vec![0.2; 384]);
        let mem3_id = mem3.id.clone();

        // Store all memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.storage.store_memory(&mem3).unwrap();

        // Add to indexes
        {
            let mut index = planner.vector_index.write().unwrap();
            index.add(mem1_id.clone(), &mem1.embedding).unwrap();
            index.add(mem2_id.clone(), &mem2.embedding).unwrap();
            index.add(mem3_id.clone(), &mem3.embedding).unwrap();
        }
        planner
            .temporal_index
            .add(&mem1_id, mem1.created_at)
            .unwrap();
        planner
            .temporal_index
            .add(&mem2_id, mem2.created_at)
            .unwrap();
        planner
            .temporal_index
            .add(&mem3_id, mem3.created_at)
            .unwrap();

        // Query with ns1 filter
        let (_, results, _) = planner
            .query("test", &vec![0.1; 384], 10, Some("ns1"), None)
            .unwrap();

        // Should only contain mem1
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.id.to_u64() == mem1_id.to_u64()));

        // Query with ns2 filter
        let (_, results, _) = planner
            .query("test", &vec![0.1; 384], 10, Some("ns2"), None)
            .unwrap();

        // Should only contain mem2
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.id.to_u64() == mem2_id.to_u64()));

        // Query with default namespace filter
        let (_, results, _) = planner
            .query("test", &vec![0.1; 384], 10, Some(""), None)
            .unwrap();

        // Should only contain mem3
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.id.to_u64() == mem3_id.to_u64()));
    }

    #[test]
    fn test_temporal_range_query_with_namespace() {
        let (planner, _dir) = create_test_planner();

        let now = Timestamp::now();
        let ts1 = now.subtract_days(1);
        // Use slightly different timestamp for mem2 to avoid any potential uniqueness issues
        let ts2 = Timestamp::from_unix_secs(ts1.as_unix_secs() + 60.0); // 1 minute later

        // Create memories in different namespaces
        let mut mem1 = Memory::new_with_timestamp("NS1 memory".to_string(), vec![0.1; 384], ts1);
        mem1.set_namespace("ns1");
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new_with_timestamp("NS2 memory".to_string(), vec![0.2; 384], ts2);
        mem2.set_namespace("ns2");
        let mem2_id = mem2.id.clone();

        // Store and index
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner
            .temporal_index
            .add(&mem1_id, mem1.created_at)
            .unwrap();
        planner
            .temporal_index
            .add(&mem2_id, mem2.created_at)
            .unwrap();

        // First verify memories are stored correctly
        let retrieved_mem1 = planner.storage.get_memory(&mem1_id).unwrap().unwrap();
        assert_eq!(retrieved_mem1.get_namespace(), "ns1");

        // Query without namespace filter first
        let all_results = planner
            .temporal_range_query(now.subtract_days(2), now, 10, None)
            .unwrap();
        assert_eq!(
            all_results.len(),
            2,
            "Should find both memories without filter"
        );

        // Query with namespace filter
        let results = planner
            .temporal_range_query(now.subtract_days(2), now, 10, Some("ns1"))
            .unwrap();

        // Should only return mem1
        assert_eq!(results.len(), 1, "Should find exactly one memory in ns1");
        assert_eq!(results[0].id, mem1_id);
    }

    #[test]
    fn test_filter_by_namespace() {
        let (planner, _dir) = create_test_planner();

        // Create memories in different namespaces
        let mut mem1 = Memory::new("NS1 memory".to_string(), vec![0.1; 384]);
        mem1.set_namespace("ns1");
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("NS2 memory".to_string(), vec![0.2; 384]);
        mem2.set_namespace("ns2");
        let mem2_id = mem2.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);

        // Filter by ns1
        planner.filter_by_namespace(&mut scores, "ns1").unwrap();

        // Should only contain mem1
        assert_eq!(scores.len(), 1);
        assert!(scores.contains_key(&mem1_id));
        assert!(!scores.contains_key(&mem2_id));
    }

    #[test]
    fn test_filter_by_metadata_exact_match() {
        let (planner, _dir) = create_test_planner();

        // Create memories with different metadata
        let mut mem1 = Memory::new("Event memory".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        mem1.metadata
            .insert("priority".to_string(), "high".to_string());
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("Task memory".to_string(), vec![0.2; 384]);
        mem2.metadata.insert("type".to_string(), "task".to_string());
        mem2.metadata
            .insert("priority".to_string(), "low".to_string());
        let mem2_id = mem2.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);

        // Filter by type=event
        let filters = vec![MetadataFilter::eq("type", "event")];
        planner.filter_by_metadata(&mut scores, &filters).unwrap();

        // Should only contain mem1
        assert_eq!(scores.len(), 1);
        assert!(scores.contains_key(&mem1_id));
        assert!(!scores.contains_key(&mem2_id));
    }

    #[test]
    fn test_filter_by_metadata_multiple_filters() {
        let (planner, _dir) = create_test_planner();

        // Create memories with different metadata
        let mut mem1 = Memory::new("High priority event".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        mem1.metadata
            .insert("priority".to_string(), "high".to_string());
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("Low priority event".to_string(), vec![0.2; 384]);
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        mem2.metadata
            .insert("priority".to_string(), "low".to_string());
        let mem2_id = mem2.id.clone();

        let mut mem3 = Memory::new("High priority task".to_string(), vec![0.3; 384]);
        mem3.metadata.insert("type".to_string(), "task".to_string());
        mem3.metadata
            .insert("priority".to_string(), "high".to_string());
        let mem3_id = mem3.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.storage.store_memory(&mem3).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);
        scores.insert(mem3_id.clone(), 0.7);

        // Filter by type=event AND priority=high
        let filters = vec![
            MetadataFilter::eq("type", "event"),
            MetadataFilter::eq("priority", "high"),
        ];
        planner.filter_by_metadata(&mut scores, &filters).unwrap();

        // Should only contain mem1 (only event with high priority)
        assert_eq!(scores.len(), 1);
        assert!(scores.contains_key(&mem1_id));
    }

    #[test]
    fn test_filter_by_metadata_comparison_operators() {
        let (planner, _dir) = create_test_planner();

        // Create memories with numeric priority values
        let mut mem1 = Memory::new("Priority 8".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("priority".to_string(), "8".to_string());
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("Priority 5".to_string(), vec![0.2; 384]);
        mem2.metadata
            .insert("priority".to_string(), "5".to_string());
        let mem2_id = mem2.id.clone();

        let mut mem3 = Memory::new("Priority 3".to_string(), vec![0.3; 384]);
        mem3.metadata
            .insert("priority".to_string(), "3".to_string());
        let mem3_id = mem3.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.storage.store_memory(&mem3).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);
        scores.insert(mem3_id.clone(), 0.7);

        // Filter by priority >= "5"
        let filters = vec![MetadataFilter::gte("priority", "5")];
        planner.filter_by_metadata(&mut scores, &filters).unwrap();

        // Should contain mem1 and mem2 (priority 8 and 5)
        assert_eq!(scores.len(), 2);
        assert!(scores.contains_key(&mem1_id));
        assert!(scores.contains_key(&mem2_id));
        assert!(!scores.contains_key(&mem3_id));
    }

    #[test]
    fn test_filter_by_metadata_in_operator() {
        let (planner, _dir) = create_test_planner();

        // Create memories with different categories
        let mut mem1 = Memory::new("Food memory".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("category".to_string(), "food".to_string());
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("Travel memory".to_string(), vec![0.2; 384]);
        mem2.metadata
            .insert("category".to_string(), "travel".to_string());
        let mem2_id = mem2.id.clone();

        let mut mem3 = Memory::new("Work memory".to_string(), vec![0.3; 384]);
        mem3.metadata
            .insert("category".to_string(), "work".to_string());
        let mem3_id = mem3.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.storage.store_memory(&mem3).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);
        scores.insert(mem3_id.clone(), 0.7);

        // Filter by category IN ["food", "travel"]
        let filters = vec![MetadataFilter::in_list(
            "category",
            vec!["food".to_string(), "travel".to_string()],
        )];
        planner.filter_by_metadata(&mut scores, &filters).unwrap();

        // Should contain mem1 and mem2
        assert_eq!(scores.len(), 2);
        assert!(scores.contains_key(&mem1_id));
        assert!(scores.contains_key(&mem2_id));
        assert!(!scores.contains_key(&mem3_id));
    }

    #[test]
    fn test_memory_matches_filters() {
        // Test exact match
        let mut memory = Memory::new("Test".to_string(), vec![0.1; 384]);
        memory
            .metadata
            .insert("type".to_string(), "event".to_string());

        let filters = vec![MetadataFilter::eq("type", "event")];
        assert!(QueryPlanner::memory_matches_filters(&memory, &filters));

        let filters = vec![MetadataFilter::eq("type", "task")];
        assert!(!QueryPlanner::memory_matches_filters(&memory, &filters));

        // Test missing field
        let filters = vec![MetadataFilter::eq("priority", "high")];
        assert!(!QueryPlanner::memory_matches_filters(&memory, &filters));

        // Test multiple filters (AND logic)
        memory
            .metadata
            .insert("priority".to_string(), "high".to_string());
        let filters = vec![
            MetadataFilter::eq("type", "event"),
            MetadataFilter::eq("priority", "high"),
        ];
        assert!(QueryPlanner::memory_matches_filters(&memory, &filters));

        let filters = vec![
            MetadataFilter::eq("type", "event"),
            MetadataFilter::eq("priority", "low"),
        ];
        assert!(!QueryPlanner::memory_matches_filters(&memory, &filters));
    }

    #[test]
    fn test_temporal_content_matching() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        // Create ingestion pipeline to extract temporal expressions
        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false, // Disable entity extraction for this test
        );

        // Add memories with temporal expressions
        let mem1 = Memory::new(
            "We had a meeting yesterday about the project".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        let mem2 = Memory::new(
            "The conference was on June 15th, 2023".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        let mem3 = Memory::new(
            "Machine learning is a fascinating field".to_string(), // No temporal expression
            vec![0.3; 384],
        );
        let mem3_id = mem3.id.clone();
        pipeline.add(mem3).unwrap();

        // Query with temporal expression matching mem1
        let scores = planner
            .temporal_search("What happened yesterday?", 10)
            .unwrap();

        // Should find mem1 (has "yesterday")
        assert!(scores.contains_key(&mem1_id), "Should find memory with 'yesterday'");

        // mem1 should have higher score than mem3 (no temporal expression)
        let score1 = scores.get(&mem1_id).unwrap_or(&0.0);
        let score3 = scores.get(&mem3_id).unwrap_or(&0.0);
        assert!(
            score1 > score3,
            "Memory with matching temporal expression should score higher than non-temporal memory"
        );
    }

    #[test]
    fn test_temporal_fallback_to_recency() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memories without temporal expressions
        let mem1 = Memory::new(
            "Machine learning techniques".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        let mem2 = Memory::new(
            "Deep learning models".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        // Query without temporal expression
        let scores = planner
            .temporal_search("Tell me about AI", 10)
            .unwrap();

        // Should fall back to weak recency signal
        // Both memories should have scores (recency fallback)
        assert!(scores.contains_key(&mem1_id));
        assert!(scores.contains_key(&mem2_id));

        // Scores should be weak (0.0-0.3 range)
        let score1 = scores.get(&mem1_id).unwrap();
        let score2 = scores.get(&mem2_id).unwrap();
        assert!(
            *score1 <= 0.31 && *score2 <= 0.31,
            "Fallback scores should be weak (≤ 0.3)"
        );

        // More recent memory should have slightly higher score
        assert!(
            score2 >= score1,
            "More recent memory should have higher fallback score"
        );
    }

    #[test]
    fn test_temporal_fallback_spread_timestamps() {
        // Tests the PRODUCTION path: memories spread across months/years.
        // Half-life decay should dominate — recent memory scores much higher
        // than old memory, regardless of rank position.
        use crate::ingest::IngestionPipeline;
        use crate::types::Timestamp;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Memory from 2 years ago
        let old_ts = Timestamp::now().subtract_days(730);
        let mem_old = Memory::new_with_timestamp(
            "Old conversation about cooking".to_string(),
            vec![0.1; 384],
            old_ts,
        );
        let old_id = mem_old.id.clone();
        pipeline.add(mem_old).unwrap();

        // Memory from 30 days ago
        let mid_ts = Timestamp::now().subtract_days(30);
        let mem_mid = Memory::new_with_timestamp(
            "Recent discussion about travel".to_string(),
            vec![0.2; 384],
            mid_ts,
        );
        let mid_id = mem_mid.id.clone();
        pipeline.add(mem_mid).unwrap();

        // Memory from today
        let mem_new = Memory::new(
            "Just talked about music".to_string(),
            vec![0.3; 384],
        );
        let new_id = mem_new.id.clone();
        pipeline.add(mem_new).unwrap();

        let scores = planner
            .temporal_search("What have we discussed?", 10)
            .unwrap();

        let score_old = *scores.get(&old_id).unwrap();
        let score_mid = *scores.get(&mid_id).unwrap();
        let score_new = *scores.get(&new_id).unwrap();

        // Ordering: new > mid > old
        assert!(
            score_new > score_mid,
            "Today's memory ({}) should score higher than 30-day old ({})",
            score_new, score_mid
        );
        assert!(
            score_mid > score_old,
            "30-day old memory ({}) should score higher than 2-year old ({})",
            score_mid, score_old
        );

        // 2-year old memory should have decayed significantly
        // Half-life 365 days → 2 years = 2 half-lives → decay ~0.25
        // Score should be much less than new memory
        assert!(
            score_new > score_old * 2.0,
            "New memory ({}) should be at least 2x the 2-year old ({})",
            score_new, score_old
        );

        // All scores in weak range
        assert!(score_new <= 0.31, "Scores should stay in 0-0.3 range, got {}", score_new);
    }

    #[test]
    fn test_temporal_fallback_same_timestamps_rank_tiebreaker() {
        // Tests the BENCHMARK path: memories with nearly identical timestamps.
        // Rank tiebreaker should ensure ordering is preserved even when
        // half-life scores are nearly uniform.
        use crate::ingest::IngestionPipeline;
        use crate::types::Timestamp;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // All memories created within milliseconds (simulates batch ingestion)
        let base_ts = Timestamp::now().subtract_days(100);
        for i in 0..5 {
            // Tiny offset: 1ms apart
            let ts = Timestamp::from_micros(base_ts.as_micros() + (i as u64 * 1000));
            let mem = Memory::new_with_timestamp(
                format!("Memory number {}", i),
                vec![0.1 * (i as f32 + 1.0); 384],
                ts,
            );
            pipeline.add(mem).unwrap();
        }

        let scores = planner
            .temporal_search("Tell me something", 10)
            .unwrap();

        // All 5 memories should be present
        assert_eq!(scores.len(), 5, "All 5 memories should have scores");

        // Collect scores sorted by value (descending)
        let mut score_vals: Vec<f32> = scores.values().copied().collect();
        score_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Scores should be strictly decreasing (rank tiebreaker works)
        for i in 0..score_vals.len() - 1 {
            assert!(
                score_vals[i] > score_vals[i + 1],
                "Score {} ({}) should be > score {} ({}): rank tiebreaker should differentiate",
                i, score_vals[i], i + 1, score_vals[i + 1]
            );
        }

        // Score spread should be small but nonzero (half-life is uniform,
        // only the 4% rank tiebreaker differentiates)
        let spread = score_vals[0] - score_vals[score_vals.len() - 1];
        assert!(
            spread > 0.001,
            "Score spread should be > 0.001 from rank tiebreaker, got {}",
            spread
        );
        assert!(
            spread < 0.1,
            "Score spread should be modest (< 0.1), got {}",
            spread
        );
    }

    #[test]
    fn test_temporal_content_matching_absolute_dates() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memory with absolute date
        let mem1 = Memory::new(
            "The conference was scheduled for June 15th, 2023".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        let mem2 = Memory::new(
            "We met in May 2024 to discuss plans".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        // Query with date reference
        let scores = planner
            .temporal_search("When was the June conference?", 10)
            .unwrap();

        // Should find mem1 (has "June")
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with June date"
        );

        // mem1 should score higher than mem2 (different month)
        let score1 = scores.get(&mem1_id).unwrap_or(&0.0);
        let score2 = scores.get(&mem2_id).unwrap_or(&0.0);
        assert!(
            score1 >= score2,
            "Memory with matching month should score higher or equal to non-matching month"
        );
    }

    #[test]
    fn test_entity_content_matching() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        // Create entity profiles so profile-based detection can find them
        planner.storage.store_entity_profile(
            &crate::types::EntityProfile::new(
                crate::types::EntityId::new(), "Alice".into(), "person".into(),
            ),
        ).unwrap();

        // Create ingestion pipeline with entity extraction enabled
        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            true, // Enable entity extraction
        );

        // Add memories with entities
        let mem1 = Memory::new(
            "Alice presented Project Alpha at the conference".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        let mem2 = Memory::new(
            "Bob worked on Project Beta for Acme Corp".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        let mem3 = Memory::new(
            "Machine learning is fascinating".to_string(), // No proper entities
            vec![0.3; 384],
        );
        let mem3_id = mem3.id.clone();
        pipeline.add(mem3).unwrap();

        // Query with entity matching mem1
        let scores = planner
            .entity_search("Tell me about Alice and Project Alpha", 10)
            .unwrap();

        // Should find mem1 (has Alice)
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with matching entities"
        );

        // mem1 should have higher score than mem3 (no entities)
        let score1 = scores.get(&mem1_id).unwrap_or(&0.0);
        let score3 = scores.get(&mem3_id).unwrap_or(&0.0);
        assert!(
            score1 > score3,
            "Memory with matching entities should score higher than non-entity memory"
        );
    }

    #[test]
    fn test_entity_search_filters_stop_words() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        // Create entity profile for Alice
        planner.storage.store_entity_profile(
            &crate::types::EntityProfile::new(
                crate::types::EntityId::new(), "Alice".into(), "person".into(),
            ),
        ).unwrap();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            true,
        );

        // Add memory with entities
        let mem1 = Memory::new(
            "Alice met Bob at Building C".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Query with stop words - should filter them out
        // "The" and "What" are not entity profile names
        let scores = planner
            .entity_search("What about Alice and The meeting?", 10)
            .unwrap();

        // Should find mem1 (Alice is a stored entity profile)
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with Alice, ignoring stop words"
        );
    }

    #[test]
    fn test_entity_search_empty_when_no_entities() {
        let (planner, _dir) = create_test_planner();

        // Query without entities (all lowercase, no proper nouns)
        let scores = planner
            .entity_search("tell me about machine learning", 10)
            .unwrap();

        // Should return empty (no entities in query)
        assert_eq!(
            scores.len(),
            0,
            "Should return empty when query has no entities"
        );
    }

    #[test]
    fn test_entity_search_multi_word_entity() {
        use crate::ingest::IngestionPipeline;
        use crate::types::{EntityFact, EntityId, EntityProfile};

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            true,
        );

        // Add memory with multi-word entity
        let mem1 = Memory::new(
            "Project Alpha is managed by the team".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Create entity profile for "Project Alpha"
        let profile = EntityProfile {
            entity_id: EntityId::new(),
            name: "Project Alpha".to_string(),
            entity_type: "project".to_string(),
            facts: std::collections::HashMap::new(),
            source_memories: vec![mem1_id.clone()],
            updated_at: crate::types::Timestamp::now(),
            summary: None,
        };
        planner.storage.store_entity_profile(&profile).unwrap();

        // Query with full multi-word entity name
        let scores = planner.entity_search("What is Project Alpha doing?", 10).unwrap();

        // Should find mem1 (exact multi-word match: "Project Alpha")
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with multi-word entity match"
        );
    }

    #[test]
    fn test_entity_overlap_calculation() {
        // Test exact match
        let query_set: std::collections::HashSet<String> =
            vec!["alice".to_string(), "bob".to_string()]
                .into_iter()
                .collect();

        let memory_entities = vec!["Alice".to_string(), "Bob".to_string()];

        let score =
            QueryPlanner::calculate_entity_overlap(&query_set, &memory_entities);

        // Both entities match exactly (case-insensitive)
        assert!((score - 1.0).abs() < 0.01, "Exact match should score 1.0");

        // Test partial match
        let query_set2: std::collections::HashSet<String> =
            vec!["alice".to_string()].into_iter().collect();

        let memory_entities2 = vec!["Bob".to_string()];

        let score2 =
            QueryPlanner::calculate_entity_overlap(&query_set2, &memory_entities2);

        // No match
        assert_eq!(score2, 0.0, "No match should score 0.0");

        // Test substring match
        let query_set3: std::collections::HashSet<String> =
            vec!["project".to_string()].into_iter().collect();

        let memory_entities3 = vec!["Project Alpha".to_string()];

        let score3 =
            QueryPlanner::calculate_entity_overlap(&query_set3, &memory_entities3);

        // Substring match should score 0.5
        assert!(
            (score3 - 0.5).abs() < 0.01,
            "Substring match should score 0.5"
        );
    }

    #[test]
    fn test_causal_content_matching() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memory with causal language
        let mem1 = Memory::new(
            "The meeting was cancelled because Alice was sick".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Add memory with high causal density
        let mem2 = Memory::new(
            "The bug was caused by a race condition which led to crashes".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        // Add memory without causal language
        let mem3 = Memory::new(
            "We had a nice lunch today".to_string(),
            vec![0.3; 384],
        );
        let mem3_id = mem3.id.clone();
        pipeline.add(mem3).unwrap();

        // Query with causal intent
        let scores = planner
            .causal_search("Why was the meeting cancelled?", 10)
            .unwrap();

        // Should find mem1 and mem2 (have causal language)
        assert!(
            scores.contains_key(&mem1_id) || scores.contains_key(&mem2_id),
            "Should find memories with causal language"
        );

        // mem3 should not be in results (no causal language)
        assert!(
            !scores.contains_key(&mem3_id) || scores.get(&mem3_id).unwrap_or(&1.0) < &0.01,
            "Should not score non-causal memory highly"
        );
    }

    #[test]
    fn test_causal_search_empty_when_no_intent() {
        let (planner, _dir) = create_test_planner();

        // Query without causal intent
        let scores = planner
            .causal_search("Tell me about machine learning", 10)
            .unwrap();

        // Should return empty (no causal intent)
        assert_eq!(
            scores.len(),
            0,
            "Should return empty when query has no causal intent"
        );
    }

    #[test]
    fn test_causal_search_with_causal_intent() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memory with causal explanation
        let mem1 = Memory::new(
            "The server crashed because of a memory leak. This was caused by unclosed connections.".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Query with "why" - clear causal intent
        let scores = planner
            .causal_search("Why did the server crash?", 10)
            .unwrap();

        // Should find mem1 (has causal explanation)
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with causal explanation for 'why' question"
        );

        // Score should be significant
        let score = scores.get(&mem1_id).unwrap();
        assert!(
            *score > 0.0,
            "Causal memory should have positive score"
        );
    }

    #[test]
    fn test_causal_search_density_threshold() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memory with very low causal density (one marker in long text)
        let mem1 = Memory::new(
            "This is a very long memory with lots of words that do not have causal markers except for this one because word.".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Query with causal intent
        let scores = planner
            .causal_search("Why did this happen?", 10)
            .unwrap();

        // Should not find mem1 (causal density too low, < 0.1 threshold)
        assert!(
            !scores.contains_key(&mem1_id),
            "Should filter out memories with very low causal density"
        );
    }

    // Sprint 18 Task 18.4: Entity-first retrieval path tests
    #[test]
    fn test_entity_focused_retrieval() {
        use crate::ingest::IngestionPipeline;
        use crate::Entity;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            true, // Enable entity extraction
        );

        // Create and store entity
        let entity = Entity::new("Alice");
        planner.storage.store_entity(&entity).unwrap();

        // Add memories mentioning Alice (scattered across turns)
        let mem1 = Memory::new(
            "Alice likes playing tennis on weekends".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        let mem2 = Memory::new(
            "Alice enjoys reading science fiction books".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        let mem3 = Memory::new(
            "Alice prefers coffee over tea".to_string(),
            vec![0.3; 384],
        );
        let mem3_id = mem3.id.clone();
        pipeline.add(mem3).unwrap();

        // Unrelated memory
        let mem4 = Memory::new(
            "Bob works on machine learning projects".to_string(),
            vec![0.4; 384],
        );
        pipeline.add(mem4).unwrap();

        // Link memories to entity manually (since entity extraction is being tested separately)
        {
            let mut graph = planner.graph_manager.write().unwrap();
            graph.link_memory_to_entity(&mem1_id, &entity.id);
            graph.link_memory_to_entity(&mem2_id, &entity.id);
            graph.link_memory_to_entity(&mem3_id, &entity.id);
        }

        // Query with entity-focused pattern (should trigger entity-first retrieval)
        let query_embedding = vec![0.15; 384];
        let (intent, results, _matched_facts) = planner
            .query("What does Alice like?", &query_embedding, 10, None, None)
            .unwrap();

        // Should classify as Entity intent
        assert_eq!(intent.intent, crate::query::QueryIntent::Entity);

        // Should extract "Alice" as entity_focus
        assert_eq!(intent.entity_focus, Some("Alice".to_string()));

        // Should retrieve ALL memories mentioning Alice (not limited by top-K semantic)
        assert!(results.len() >= 3, "Should retrieve all memories mentioning Alice");

        // Verify all Alice memories are in results
        let result_ids: Vec<_> = results.iter().map(|r| &r.id).collect();
        assert!(result_ids.contains(&&mem1_id), "Should include mem1 (tennis)");
        assert!(result_ids.contains(&&mem2_id), "Should include mem2 (books)");
        assert!(result_ids.contains(&&mem3_id), "Should include mem3 (coffee)");

        // All results should have high entity_score (1.0 since they mention the entity)
        for result in &results {
            if result_ids.contains(&&result.id) {
                assert_eq!(
                    result.entity_score, 1.0,
                    "All entity-focused results should have entity_score=1.0"
                );
            }
        }
    }

    #[test]
    fn test_entity_focused_retrieval_not_found() {
        let (planner, _dir) = create_test_planner();

        // Query for non-existent entity
        let query_embedding = vec![0.1; 384];
        let (_intent, results, _matched_facts) = planner
            .query("What does NonExistentEntity like?", &query_embedding, 10, None, None)
            .unwrap();

        // Should return empty results
        assert_eq!(results.len(), 0, "Should return empty for non-existent entity");
    }

    // --- extract_query_entity tests (profile-aware) ---

    #[test]
    fn test_extract_query_entity_single_name() {
        let (planner, _dir) = create_test_planner();

        // Store profiles for known entities
        planner.storage.store_entity_profile(
            &crate::types::EntityProfile::new(
                crate::types::EntityId::new(), "Melanie".into(), "person".into(),
            ),
        ).unwrap();
        planner.storage.store_entity_profile(
            &crate::types::EntityProfile::new(
                crate::types::EntityId::new(), "Caroline".into(), "person".into(),
            ),
        ).unwrap();

        assert_eq!(
            planner.extract_query_entity("What books has Melanie read?"),
            Some("melanie".to_string())
        );
        assert_eq!(
            planner.extract_query_entity("What did Caroline research?"),
            Some("caroline".to_string())
        );
    }

    #[test]
    fn test_extract_query_entity_possessive() {
        let (planner, _dir) = create_test_planner();
        planner.storage.store_entity_profile(
            &crate::types::EntityProfile::new(
                crate::types::EntityId::new(), "Caroline".into(), "person".into(),
            ),
        ).unwrap();

        assert_eq!(
            planner.extract_query_entity("What is Caroline's relationship status?"),
            Some("caroline".to_string())
        );
    }

    #[test]
    fn test_extract_query_entity_two_names_returns_none() {
        let (planner, _dir) = create_test_planner();
        planner.storage.store_entity_profile(
            &crate::types::EntityProfile::new(
                crate::types::EntityId::new(), "Caroline".into(), "person".into(),
            ),
        ).unwrap();
        planner.storage.store_entity_profile(
            &crate::types::EntityProfile::new(
                crate::types::EntityId::new(), "Melanie".into(), "person".into(),
            ),
        ).unwrap();

        // Queries about multiple people should not trigger speaker filtering
        assert_eq!(
            planner.extract_query_entity(
                "When did Caroline and Melanie go to a pride festival together?"
            ),
            None
        );
    }

    #[test]
    fn test_extract_query_entity_no_entity() {
        let (planner, _dir) = create_test_planner();
        // No profiles stored — generic queries should return None
        assert_eq!(
            planner.extract_query_entity("What happened yesterday?"),
            None
        );
    }

    #[test]
    fn test_extract_query_entity_lowercase_query() {
        let (planner, _dir) = create_test_planner();
        planner.storage.store_entity_profile(
            &crate::types::EntityProfile::new(
                crate::types::EntityId::new(), "Caroline".into(), "person".into(),
            ),
        ).unwrap();

        // Lowercase name in query should still match
        assert_eq!(
            planner.extract_query_entity("what does caroline like?"),
            Some("caroline".to_string())
        );
    }

    // --- Adaptive-K (Top-p) tests ---

    fn make_fused_result(score: f32) -> FusedResult {
        FusedResult {
            id: crate::types::MemoryId::new(),
            semantic_score: score,
            bm25_score: 0.0,
            temporal_score: 0.0,
            causal_score: 0.0,
            entity_score: 0.0,
            fused_score: score,
            confidence: 1.0,
        }
    }

    #[test]
    fn test_adaptive_k_concentrated_scores() {
        // One dominant result + many weak ones → should select fewer
        let mut results = vec![make_fused_result(5.0)];
        for _ in 0..24 {
            results.push(make_fused_result(0.1));
        }
        let k = adaptive_k_select(&results, 0.7, 25);
        // The dominant result has almost all probability mass
        // Should select close to min_k (8) since top-1 already exceeds 0.7
        assert!(k <= 10, "Expected few results but got {}", k);
        assert!(k >= 5, "Expected at least min_k=5 but got {}", k); // min_k floor
    }

    #[test]
    fn test_adaptive_k_uniform_scores() {
        // All equal scores → softmax is uniform → need many to reach 0.7
        let results: Vec<FusedResult> = (0..25).map(|_| make_fused_result(1.0)).collect();
        let k = adaptive_k_select(&results, 0.7, 25);
        // Uniform: need 70% of 25 = 18 results to reach threshold
        assert!(k >= 17, "Expected ~18 results for uniform but got {}", k);
        assert!(k <= 20, "Expected ~18 results for uniform but got {}", k);
    }

    #[test]
    fn test_adaptive_k_respects_min_k() {
        // Even with extreme concentration, min_k is respected
        let mut results = vec![make_fused_result(100.0)];
        for _ in 0..24 {
            results.push(make_fused_result(0.001));
        }
        let k = adaptive_k_select(&results, 0.7, 25);
        // limit=25, min_k = max(25/3, 5) = max(8, 5) = 8
        assert!(k >= 8, "Expected at least min_k=8 but got {}", k);
    }

    #[test]
    fn test_adaptive_k_single_result() {
        let results = vec![make_fused_result(1.0)];
        let k = adaptive_k_select(&results, 0.7, 25);
        assert_eq!(k, 1);
    }

    #[test]
    fn test_adaptive_k_empty_results() {
        let results: Vec<FusedResult> = vec![];
        let k = adaptive_k_select(&results, 0.7, 25);
        assert_eq!(k, 0);
    }

    #[test]
    fn test_adaptive_k_gradual_dropoff() {
        // Scores that drop off gradually: [1.0, 0.9, 0.8, ..., 0.1]
        let results: Vec<FusedResult> = (0..10)
            .map(|i| make_fused_result(1.0 - i as f32 * 0.1))
            .collect();
        let k = adaptive_k_select(&results, 0.7, 25);
        // Gradual dropoff: top results have more probability but not extremely so
        // Should keep most results since scores are close
        assert!(k >= 5, "Expected reasonable k for gradual dropoff but got {}", k);
    }
}
