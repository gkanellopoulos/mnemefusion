//! Python bindings for MnemeFusion
//!
//! This module provides a Pythonic interface to the MnemeFusion memory engine.

use mnemefusion_core::{
    types::{BatchResult, MemoryInput, Source, SourceType},
    Config, MemoryEngine, MemoryId, Timestamp,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

/// Helper function to parse Source from Python dict
fn parse_source_from_dict(dict: &PyDict) -> PyResult<Source> {
    let source_type_str: String = dict
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("Source 'type' is required"))?
        .extract()?;

    let source_type = SourceType::from_str(&source_type_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid source type: {}", e)))?;

    let mut source = Source::new(source_type);

    if let Some(id) = dict.get_item("id")? {
        source = source.with_id(id.extract::<String>()?);
    }
    if let Some(location) = dict.get_item("location")? {
        source = source.with_location(location.extract::<String>()?);
    }
    if let Some(timestamp) = dict.get_item("timestamp")? {
        source = source.with_timestamp(timestamp.extract::<String>()?);
    }
    if let Some(original_text) = dict.get_item("original_text")? {
        source = source.with_original_text(original_text.extract::<String>()?);
    }
    if let Some(confidence) = dict.get_item("confidence")? {
        source = source.with_confidence(confidence.extract::<f32>()?);
    }
    if let Some(extractor) = dict.get_item("extractor")? {
        source = source.with_extractor(extractor.extract::<String>()?);
    }
    if let Some(metadata) = dict.get_item("metadata")? {
        source = source.with_metadata(metadata.extract::<HashMap<String, String>>()?);
    }

    Ok(source)
}

/// Helper function to convert Source to Python dict
fn source_to_pydict(py: Python, source: &Source) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("type", source.source_type.to_string())?;

    if let Some(id) = &source.id {
        dict.set_item("id", id)?;
    }
    if let Some(location) = &source.location {
        dict.set_item("location", location)?;
    }
    if let Some(timestamp) = &source.timestamp {
        dict.set_item("timestamp", timestamp)?;
    }
    if let Some(original_text) = &source.original_text {
        dict.set_item("original_text", original_text)?;
    }
    if let Some(confidence) = source.confidence {
        dict.set_item("confidence", confidence)?;
    }
    if let Some(extractor) = &source.extractor {
        dict.set_item("extractor", extractor)?;
    }
    if let Some(metadata) = &source.metadata {
        dict.set_item("metadata", metadata)?;
    }

    Ok(dict.into())
}

/// Python wrapper for MemoryEngine
#[pyclass(name = "Memory")]
pub struct PyMemory {
    engine: RefCell<Option<MemoryEngine>>,
}

impl PyMemory {
    /// Helper to get a reference to the engine, returning error if closed
    fn get_engine(&self) -> PyResult<std::cell::Ref<MemoryEngine>> {
        let borrow = self.engine.borrow();
        if borrow.is_none() {
            return Err(PyRuntimeError::new_err("Database is closed"));
        }
        Ok(std::cell::Ref::map(borrow, |opt| {
            opt.as_ref().unwrap()
        }))
    }
}

#[pymethods]
impl PyMemory {
    /// Create or open a memory database
    ///
    /// Args:
    ///     path: Path to the .mfdb file
    ///     config: Optional configuration dictionary
    ///
    /// Returns:
    ///     A new Memory instance
    ///
    /// Example:
    ///     >>> memory = Memory("brain.mfdb")
    ///     >>> memory = Memory("brain.mfdb", config={"embedding_dim": 384})
    #[new]
    #[pyo3(signature = (path, config=None))]
    fn new(path: &str, config: Option<&PyDict>) -> PyResult<Self> {
        let mut rust_config = Config::default();

        if let Some(cfg) = config {
            if let Some(dim) = cfg.get_item("embedding_dim")? {
                rust_config.embedding_dim = dim.extract()?;
            }
            if let Some(entity_extraction) = cfg.get_item("entity_extraction_enabled")? {
                rust_config.entity_extraction_enabled = entity_extraction.extract()?;
            }
        }

        let engine = MemoryEngine::open(PathBuf::from(path), rust_config)
            .map_err(|e| PyIOError::new_err(format!("Failed to open database: {}", e)))?;

        Ok(Self {
            engine: RefCell::new(Some(engine)),
        })
    }

    /// Add a new memory to the database
    ///
    /// Args:
    ///     content: Text content to store
    ///     embedding: Vector embedding (list of floats)
    ///     metadata: Optional metadata dictionary
    ///     timestamp: Optional Unix timestamp (seconds since epoch)
    ///     source: Optional source/provenance tracking dictionary
    ///
    /// Returns:
    ///     Memory ID as a string
    ///
    /// Example:
    ///     >>> embedding = [0.1] * 384
    ///     >>> memory_id = memory.add("Meeting notes", embedding)
    ///     >>> memory_id = memory.add("Meeting notes", embedding, metadata={"project": "Alpha"})
    ///     >>> source = {"type": "conversation", "id": "conv_123", "confidence": 0.95}
    ///     >>> memory_id = memory.add("Meeting notes", embedding, source=source)
    #[pyo3(signature = (content, embedding, metadata=None, timestamp=None, source=None))]
    fn add(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<f64>,
        source: Option<&PyDict>,
    ) -> PyResult<String> {
        let engine = self.get_engine()?;
        let ts = timestamp.map(|t| Timestamp::from_unix_secs(t));

        // Parse source from Python dict
        let rust_source = if let Some(src_dict) = source {
            Some(parse_source_from_dict(src_dict)?)
        } else {
            None
        };

        let id = engine
            .add(content, embedding, metadata, ts, rust_source)
            .map_err(|e| PyValueError::new_err(format!("Failed to add memory: {}", e)))?;

        Ok(id.to_string())
    }

    /// Retrieve a memory by ID
    ///
    /// Args:
    ///     memory_id: Memory ID string
    ///
    /// Returns:
    ///     Dictionary with memory data, or None if not found
    ///
    /// Example:
    ///     >>> result = memory.get(memory_id)
    ///     >>> if result:
    ///     >>>     print(result["content"])
    fn get(&self, memory_id: &str) -> PyResult<Option<PyObject>> {
        let engine = self.get_engine()?;
        let id = MemoryId::parse(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        let mem = engine
            .get(&id)
            .map_err(|e| PyIOError::new_err(format!("Failed to get memory: {}", e)))?;

        if let Some(memory) = mem {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("id", memory.id.to_string())?;
                dict.set_item("content", memory.content.clone())?;
                dict.set_item("embedding", memory.embedding.clone())?;
                dict.set_item("metadata", memory.metadata.clone())?;
                dict.set_item("created_at", memory.created_at.as_unix_secs())?;

                // Add source if present
                if let Ok(Some(source)) = memory.get_source() {
                    let source_dict = source_to_pydict(py, &source)?;
                    dict.set_item("source", source_dict)?;
                } else {
                    dict.set_item("source", py.None())?;
                }

                Ok(Some(dict.into()))
            })
        } else {
            Ok(None)
        }
    }

    /// Delete a memory by ID
    ///
    /// Args:
    ///     memory_id: Memory ID string
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn delete(&self, memory_id: &str) -> PyResult<bool> {
        let engine = self.get_engine()?;
        let id = MemoryId::parse(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        engine
            .delete(&id)
            .map_err(|e| PyIOError::new_err(format!("Failed to delete memory: {}", e)))
    }

    /// Add multiple memories in a batch operation (10x+ faster than individual adds)
    ///
    /// Args:
    ///     memories: List of dicts with memory data:
    ///         - content: Text content (required)
    ///         - embedding: Vector embedding (required)
    ///         - metadata: Optional metadata dict
    ///         - timestamp: Optional Unix timestamp
    ///         - source: Optional source tracking dict
    ///
    /// Returns:
    ///     Dictionary with results:
    ///         - ids: List of created memory IDs
    ///         - created_count: Number of memories successfully created
    ///         - duplicate_count: Number of duplicates detected
    ///         - errors: List of error dicts (index, message, memory_id)
    ///
    /// Example:
    ///     >>> memories = [
    ///     >>>     {"content": "mem 1", "embedding": [0.1] * 384},
    ///     >>>     {"content": "mem 2", "embedding": [0.2] * 384},
    ///     >>> ]
    ///     >>> result = memory.add_batch(memories)
    ///     >>> print(f"Created {result['created_count']} memories")
    fn add_batch(&self, memories: Vec<&PyDict>) -> PyResult<PyObject> {
        let engine = self.get_engine()?;

        // Convert Python dicts to MemoryInput
        let mut inputs = Vec::new();
        for mem_dict in memories {
            // Extract required fields
            let content: String = mem_dict
                .get_item("content")?
                .ok_or_else(|| PyValueError::new_err("'content' is required"))?
                .extract()?;

            let embedding: Vec<f32> = mem_dict
                .get_item("embedding")?
                .ok_or_else(|| PyValueError::new_err("'embedding' is required"))?
                .extract()?;

            // Create MemoryInput
            let mut input = MemoryInput::new(content, embedding);

            // Add optional fields
            if let Some(metadata_item) = mem_dict.get_item("metadata")? {
                let metadata: HashMap<String, String> = metadata_item.extract()?;
                input = input.with_metadata(metadata);
            }

            if let Some(timestamp_item) = mem_dict.get_item("timestamp")? {
                let ts: f64 = timestamp_item.extract()?;
                input = input.with_timestamp(Timestamp::from_unix_secs(ts));
            }

            if let Some(source_item) = mem_dict.get_item("source")? {
                let source_dict: &PyDict = source_item.downcast()?;
                let source = parse_source_from_dict(source_dict)?;
                input = input.with_source(source);
            }

            inputs.push(input);
        }

        // Call batch add
        let result: BatchResult = engine
            .add_batch(inputs)
            .map_err(|e| PyIOError::new_err(format!("Batch add failed: {}", e)))?;

        // Convert result to Python dict
        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);

            // Convert IDs to strings
            let ids: Vec<String> = result.ids.iter().map(|id| id.to_string()).collect();
            result_dict.set_item("ids", ids)?;
            result_dict.set_item("created_count", result.created_count)?;
            result_dict.set_item("duplicate_count", result.duplicate_count)?;

            // Convert errors
            let errors: Vec<PyObject> = result
                .errors
                .iter()
                .map(|err| {
                    let err_dict = PyDict::new(py);
                    err_dict.set_item("index", err.index).ok();
                    err_dict.set_item("message", &err.message).ok();
                    if let Some(ref id) = err.memory_id {
                        err_dict.set_item("memory_id", id.to_string()).ok();
                    } else {
                        err_dict.set_item("memory_id", py.None()).ok();
                    }
                    err_dict.into()
                })
                .collect();
            result_dict.set_item("errors", errors)?;

            Ok(result_dict.into())
        })
    }

    /// Delete multiple memories in a batch operation (faster than individual deletes)
    ///
    /// Args:
    ///     memory_ids: List of memory ID strings to delete
    ///
    /// Returns:
    ///     Number of memories actually deleted
    ///
    /// Example:
    ///     >>> ids = [id1, id2, id3]
    ///     >>> deleted = memory.delete_batch(ids)
    ///     >>> print(f"Deleted {deleted} memories")
    fn delete_batch(&self, memory_ids: Vec<String>) -> PyResult<usize> {
        let engine = self.get_engine()?;

        // Parse all IDs
        let ids: Result<Vec<MemoryId>, _> = memory_ids
            .iter()
            .map(|id_str| MemoryId::parse(id_str))
            .collect();

        let ids =
            ids.map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        // Call batch delete
        engine
            .delete_batch(ids)
            .map_err(|e| PyIOError::new_err(format!("Batch delete failed: {}", e)))
    }

    /// Semantic similarity search
    ///
    /// Args:
    ///     query_embedding: Query vector (list of floats)
    ///     top_k: Number of results to return
    ///
    /// Returns:
    ///     List of (memory_dict, similarity_score) tuples
    ///
    /// Example:
    ///     >>> query_embedding = [0.1] * 384
    ///     >>> results = memory.search(query_embedding, top_k=10)
    ///     >>> for mem, score in results:
    ///     >>>     print(f"{score:.3f}: {mem['content']}")
    fn search(&self, query_embedding: Vec<f32>, top_k: usize) -> PyResult<Vec<PyObject>> {
        let engine = self.get_engine()?;
        let results = engine
            .search(&query_embedding, top_k)
            .map_err(|e| PyIOError::new_err(format!("Search failed: {}", e)))?;

        Python::with_gil(|py| {
            results
                .into_iter()
                .map(|(memory, score)| {
                    let dict = PyDict::new(py);
                    dict.set_item("id", memory.id.to_string())?;
                    dict.set_item("content", memory.content.clone())?;
                    dict.set_item("embedding", memory.embedding.clone())?;
                    dict.set_item("metadata", memory.metadata.clone())?;
                    dict.set_item("created_at", memory.created_at.as_unix_secs())?;

                    // Add source if present
                    if let Ok(Some(source)) = memory.get_source() {
                        let source_dict = source_to_pydict(py, &source)?;
                        dict.set_item("source", source_dict)?;
                    } else {
                        dict.set_item("source", py.None())?;
                    }

                    let tuple = (dict, score);
                    Ok(tuple.into_py(py))
                })
                .collect()
        })
    }

    /// Intelligent multi-dimensional query with intent classification
    ///
    /// Args:
    ///     query_text: Natural language query
    ///     query_embedding: Query vector (list of floats)
    ///     limit: Maximum number of results
    ///
    /// Returns:
    ///     Tuple of (intent_dict, results_list)
    ///     - intent_dict: {"intent": str, "confidence": float}
    ///     - results_list: List of (memory_dict, scores_dict) tuples
    ///
    /// Example:
    ///     >>> intent, results = memory.query("Why was the meeting cancelled?", embedding, 10)
    ///     >>> print(f"Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
    ///     >>> for mem, scores in results:
    ///     >>>     print(f"Fused score: {scores['fused_score']:.3f}")
    ///     >>>     print(f"  Content: {mem['content']}")
    fn query(
        &self,
        query_text: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> PyResult<PyObject> {
        let engine = self.get_engine()?;
        let (intent, results) = engine
            .query(query_text, &query_embedding, limit)
            .map_err(|e| PyIOError::new_err(format!("Query failed: {}", e)))?;

        Python::with_gil(|py| {
            // Build intent dict
            let intent_dict = PyDict::new(py);
            intent_dict.set_item("intent", format!("{:?}", intent.intent))?;
            intent_dict.set_item("confidence", intent.confidence)?;

            // Build results list
            let results_list: PyResult<Vec<PyObject>> = results
                .into_iter()
                .map(|(memory, fused_result)| {
                    let mem_dict = PyDict::new(py);
                    mem_dict.set_item("id", memory.id.to_string())?;
                    mem_dict.set_item("content", memory.content.clone())?;
                    mem_dict.set_item("embedding", memory.embedding.clone())?;
                    mem_dict.set_item("metadata", memory.metadata.clone())?;
                    mem_dict.set_item("created_at", memory.created_at.as_unix_secs())?;

                    // Add source if present
                    if let Ok(Some(source)) = memory.get_source() {
                        let source_dict = source_to_pydict(py, &source)?;
                        mem_dict.set_item("source", source_dict)?;
                    } else {
                        mem_dict.set_item("source", py.None())?;
                    }

                    let scores_dict = PyDict::new(py);
                    scores_dict.set_item("semantic_score", fused_result.semantic_score)?;
                    scores_dict.set_item("temporal_score", fused_result.temporal_score)?;
                    scores_dict.set_item("causal_score", fused_result.causal_score)?;
                    scores_dict.set_item("entity_score", fused_result.entity_score)?;
                    scores_dict.set_item("fused_score", fused_result.fused_score)?;

                    let tuple = (mem_dict, scores_dict);
                    Ok(tuple.into_py(py))
                })
                .collect();

            let tuple = (intent_dict, results_list?);
            Ok(tuple.into_py(py))
        })
    }

    /// Get the number of memories in the database
    ///
    /// Returns:
    ///     Count of memories
    fn count(&self) -> PyResult<usize> {
        let engine = self.get_engine()?;
        engine
            .count()
            .map_err(|e| PyIOError::new_err(format!("Failed to count memories: {}", e)))
    }

    /// Add a causal link between two memories
    ///
    /// Args:
    ///     cause_id: Memory ID of the cause
    ///     effect_id: Memory ID of the effect
    ///     confidence: Confidence score (0.0 to 1.0)
    ///     evidence: Text explaining the causal relationship
    ///
    /// Example:
    ///     >>> memory.add_causal_link(cause_id, effect_id, 0.9, "Meeting was cancelled due to conflict")
    fn add_causal_link(
        &self,
        cause_id: &str,
        effect_id: &str,
        confidence: f32,
        evidence: String,
    ) -> PyResult<()> {
        let engine = self.get_engine()?;
        let cause = MemoryId::parse(cause_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid cause ID: {}", e)))?;
        let effect = MemoryId::parse(effect_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid effect ID: {}", e)))?;

        engine
            .add_causal_link(&cause, &effect, confidence, evidence)
            .map_err(|e| PyValueError::new_err(format!("Failed to add causal link: {}", e)))
    }

    /// Get causes of a memory (backward traversal)
    ///
    /// Args:
    ///     memory_id: Memory ID
    ///     max_hops: Maximum traversal depth
    ///
    /// Returns:
    ///     List of causal paths (each path is a list of memory IDs)
    fn get_causes(&self, memory_id: &str, max_hops: usize) -> PyResult<Vec<Vec<String>>> {
        let engine = self.get_engine()?;
        let id = MemoryId::parse(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        let result = engine
            .get_causes(&id, max_hops)
            .map_err(|e| PyIOError::new_err(format!("Failed to get causes: {}", e)))?;

        Ok(result
            .paths
            .into_iter()
            .map(|path| path.memories.into_iter().map(|m| m.to_string()).collect())
            .collect())
    }

    /// Get effects of a memory (forward traversal)
    ///
    /// Args:
    ///     memory_id: Memory ID
    ///     max_hops: Maximum traversal depth
    ///
    /// Returns:
    ///     List of causal paths (each path is a list of memory IDs)
    fn get_effects(&self, memory_id: &str, max_hops: usize) -> PyResult<Vec<Vec<String>>> {
        let engine = self.get_engine()?;
        let id = MemoryId::parse(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        let result = engine
            .get_effects(&id, max_hops)
            .map_err(|e| PyIOError::new_err(format!("Failed to get effects: {}", e)))?;

        Ok(result
            .paths
            .into_iter()
            .map(|path| path.memories.into_iter().map(|m| m.to_string()).collect())
            .collect())
    }

    /// List all entities in the database
    ///
    /// Returns:
    ///     List of entity dictionaries
    fn list_entities(&self) -> PyResult<Vec<PyObject>> {
        let engine = self.get_engine()?;
        let entities = engine
            .list_entities()
            .map_err(|e| PyIOError::new_err(format!("Failed to list entities: {}", e)))?;

        Python::with_gil(|py| {
            entities
                .into_iter()
                .map(|entity| {
                    let dict = PyDict::new(py);
                    dict.set_item("id", entity.id.to_string())?;
                    dict.set_item("name", entity.name)?;
                    dict.set_item("mention_count", entity.mention_count)?;
                    dict.set_item("metadata", entity.metadata)?;
                    Ok(dict.into())
                })
                .collect()
        })
    }

    /// Close the database and save all indexes
    ///
    /// Example:
    ///     >>> memory.close()
    fn close(&self) -> PyResult<()> {
        let engine_opt = self.engine.borrow_mut().take();
        if let Some(engine) = engine_opt {
            engine
                .close()
                .map_err(|e| PyIOError::new_err(format!("Failed to close database: {}", e)))?;
        }
        Ok(())
    }

    /// Context manager support: __enter__
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager support: __exit__
    fn __exit__(&self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// MnemeFusion Python module
#[pymodule]
fn mnemefusion(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMemory>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
