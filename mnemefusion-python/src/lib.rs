//! Python bindings for MnemeFusion
//!
//! This module provides a Pythonic interface to the MnemeFusion memory engine.

// Note: SLM feature is optional. If not enabled, SlmClassifier won't be available in Rust,
// but Python bindings will still work for basic memory operations.

use mnemefusion_core::{
    types::{BatchResult, FilterOp, MemoryInput, MetadataFilter, Source, SourceType},
    Config, EmbeddingFn, MemoryEngine, MemoryId, Timestamp,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

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

/// Helper function to parse metadata filters from Python dict
/// Format: {"field": "type", "op": "eq", "value": "event"}
/// or {"field": "priority", "op": "in", "values": ["high", "medium"]}
fn parse_filter_from_dict(dict: &PyDict) -> PyResult<MetadataFilter> {
    let field: String = dict
        .get_item("field")?
        .ok_or_else(|| PyValueError::new_err("Filter 'field' is required"))?
        .extract()?;

    let op_str: String = dict
        .get_item("op")?
        .ok_or_else(|| PyValueError::new_err("Filter 'op' is required"))?
        .extract()?;

    let op = match op_str.as_str() {
        "eq" => {
            let value: String = dict
                .get_item("value")?
                .ok_or_else(|| {
                    PyValueError::new_err("Filter 'value' is required for 'eq' operator")
                })?
                .extract()?;
            FilterOp::Eq(value)
        }
        "ne" => {
            let value: String = dict
                .get_item("value")?
                .ok_or_else(|| {
                    PyValueError::new_err("Filter 'value' is required for 'ne' operator")
                })?
                .extract()?;
            FilterOp::Ne(value)
        }
        "gt" => {
            let value: String = dict
                .get_item("value")?
                .ok_or_else(|| {
                    PyValueError::new_err("Filter 'value' is required for 'gt' operator")
                })?
                .extract()?;
            FilterOp::Gt(value)
        }
        "gte" => {
            let value: String = dict
                .get_item("value")?
                .ok_or_else(|| {
                    PyValueError::new_err("Filter 'value' is required for 'gte' operator")
                })?
                .extract()?;
            FilterOp::Gte(value)
        }
        "lt" => {
            let value: String = dict
                .get_item("value")?
                .ok_or_else(|| {
                    PyValueError::new_err("Filter 'value' is required for 'lt' operator")
                })?
                .extract()?;
            FilterOp::Lt(value)
        }
        "lte" => {
            let value: String = dict
                .get_item("value")?
                .ok_or_else(|| {
                    PyValueError::new_err("Filter 'value' is required for 'lte' operator")
                })?
                .extract()?;
            FilterOp::Lte(value)
        }
        "in" => {
            let values: Vec<String> = dict
                .get_item("values")?
                .ok_or_else(|| {
                    PyValueError::new_err("Filter 'values' (list) is required for 'in' operator")
                })?
                .extract()?;
            FilterOp::In(values)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown filter operator: {}",
                op_str
            )))
        }
    };

    Ok(MetadataFilter::new(field, op))
}

/// Helper function to parse metadata filters from Python list
fn parse_filters_from_list(filters: Option<&PyList>) -> PyResult<Option<Vec<MetadataFilter>>> {
    if let Some(filter_list) = filters {
        let mut parsed_filters = Vec::new();
        for item in filter_list.iter() {
            let dict = item
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Filter must be a dict"))?;
            parsed_filters.push(parse_filter_from_dict(dict)?);
        }
        Ok(Some(parsed_filters))
    } else {
        Ok(None)
    }
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
        Ok(std::cell::Ref::map(borrow, |opt| opt.as_ref().unwrap()))
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
    ///     >>> # Enable SLM metadata extraction at ingestion (recommended)
    ///     >>> memory = Memory("brain.mfdb", config={"use_slm": True, "slm_model_path": "/path/to/model"})
    ///     >>> # Disable SLM query classification for fast queries (default: False)
    ///     >>> memory = Memory("brain.mfdb", config={"use_slm": True, "slm_query_classification_enabled": False})
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
            if let Some(indexed_metadata) = cfg.get_item("indexed_metadata")? {
                rust_config.indexed_metadata = indexed_metadata.extract()?;
            }
            if let Some(passes) = cfg.get_item("extraction_passes")? {
                rust_config.extraction_passes = passes.extract::<usize>()?.clamp(1, 10);
            }
            if let Some(ak) = cfg.get_item("adaptive_k_threshold")? {
                rust_config.adaptive_k_threshold = ak.extract::<f32>()?.clamp(0.0, 1.0);
            }

            // SLM configuration (only available with 'slm' feature)
            #[cfg(feature = "slm")]
            {
                if let Some(use_slm) = cfg.get_item("use_slm")? {
                    let use_slm: bool = use_slm.extract()?;
                    if use_slm {
                        let mut slm_config = mnemefusion_core::slm::SlmConfig::default();

                        if let Some(model_path) = cfg.get_item("slm_model_path")? {
                            let path: String = model_path.extract()?;
                            slm_config = slm_config.with_model_path(PathBuf::from(path));
                        }

                        rust_config = rust_config.with_slm(slm_config);
                    }
                }

                if let Some(slm_extraction) = cfg.get_item("slm_metadata_extraction_enabled")? {
                    let enabled: bool = slm_extraction.extract()?;
                    rust_config = rust_config.with_slm_metadata_extraction(enabled);
                }

                if let Some(slm_query_class) = cfg.get_item("slm_query_classification_enabled")? {
                    let enabled: bool = slm_query_class.extract()?;
                    rust_config = rust_config.with_slm_query_classification(enabled);
                }
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
    ///     namespace: Optional namespace string for multi-user/multi-context isolation
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
    ///     >>> memory_id = memory.add("Meeting notes", embedding, namespace="user_123")
    #[pyo3(signature = (content, embedding, metadata=None, timestamp=None, source=None, namespace=None))]
    fn add(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<f64>,
        source: Option<&PyDict>,
        namespace: Option<&str>,
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
            .add(content, embedding, metadata, ts, rust_source, namespace)
            .map_err(|e| PyValueError::new_err(format!("Failed to add memory: {}", e)))?;

        Ok(id.to_string())
    }

    /// Retrieve a memory by ID
    ///
    /// Args:
    ///     memory_id: Memory ID string
    ///
    /// Returns:
    ///     Dictionary with memory data (including 'namespace' field), or None if not found
    ///
    /// Example:
    ///     >>> result = memory.get(memory_id)
    ///     >>> if result:
    ///     >>>     print(result["content"])
    ///     >>>     print(result["namespace"])  # Empty string if default namespace
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
                dict.set_item("namespace", memory.get_namespace())?;

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
    ///     namespace: Optional namespace filter. If provided, verifies memory is in this namespace before deleting
    ///
    /// Returns:
    ///     True if deleted, False if not found
    ///
    /// Raises:
    ///     ValueError: If namespace is provided and doesn't match memory's namespace
    ///
    /// Example:
    ///     >>> memory.delete(memory_id)  # Delete from any namespace
    ///     >>> memory.delete(memory_id, namespace="user_123")  # Verify namespace first
    #[pyo3(signature = (memory_id, namespace=None))]
    fn delete(&self, memory_id: &str, namespace: Option<&str>) -> PyResult<bool> {
        let engine = self.get_engine()?;
        let id = MemoryId::parse(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        engine
            .delete(&id, namespace)
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
    ///         - namespace: Optional namespace string
    ///     namespace: Optional namespace to apply to all memories (overridden by per-memory namespace)
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
    ///     >>>     {"content": "mem 2", "embedding": [0.2] * 384, "namespace": "user_456"},
    ///     >>> ]
    ///     >>> result = memory.add_batch(memories, namespace="user_123")
    ///     >>> print(f"Created {result['created_count']} memories")
    #[pyo3(signature = (memories, namespace=None))]
    fn add_batch(&self, memories: Vec<&PyDict>, namespace: Option<&str>) -> PyResult<PyObject> {
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

            // Handle namespace: per-memory namespace takes precedence
            if let Some(ns_item) = mem_dict.get_item("namespace")? {
                let ns: String = ns_item.extract()?;
                input = input.with_namespace(ns);
            }

            inputs.push(input);
        }

        // Call batch add with namespace parameter
        let result: BatchResult = engine
            .add_batch(inputs, namespace)
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
    ///     namespace: Optional namespace filter. Only deletes memories in this namespace
    ///
    /// Returns:
    ///     Number of memories actually deleted
    ///
    /// Example:
    ///     >>> ids = [id1, id2, id3]
    ///     >>> deleted = memory.delete_batch(ids)
    ///     >>> deleted = memory.delete_batch(ids, namespace="user_123")  # Only delete from user_123
    #[pyo3(signature = (memory_ids, namespace=None))]
    fn delete_batch(&self, memory_ids: Vec<String>, namespace: Option<&str>) -> PyResult<usize> {
        let engine = self.get_engine()?;

        // Parse all IDs
        let ids: Result<Vec<MemoryId>, _> = memory_ids
            .iter()
            .map(|id_str| MemoryId::parse(id_str))
            .collect();

        let ids = ids.map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        // Call batch delete with namespace parameter
        engine
            .delete_batch(ids, namespace)
            .map_err(|e| PyIOError::new_err(format!("Batch delete failed: {}", e)))
    }

    /// Add a memory with automatic deduplication
    ///
    /// Checks if identical content already exists. If duplicate found,
    /// returns existing ID without creating a new memory.
    ///
    /// Args:
    ///     content: Text content to store
    ///     embedding: Vector embedding (list of floats)
    ///     metadata: Optional metadata dictionary
    ///     timestamp: Optional Unix timestamp (seconds since epoch)
    ///     source: Optional source/provenance tracking dictionary
    ///     namespace: Optional namespace string for multi-user/multi-context isolation
    ///
    /// Returns:
    ///     Dictionary with results:
    ///         - id: Memory ID (either new or existing)
    ///         - created: True if new memory created, False if duplicate
    ///         - existing_id: ID of existing memory if duplicate (same as id)
    ///
    /// Example:
    ///     >>> embedding = [0.1] * 384
    ///     >>> result1 = memory.add_with_dedup("Meeting notes", embedding)
    ///     >>> print(f"Created: {result1['created']}")  # True
    ///     >>> result2 = memory.add_with_dedup("Meeting notes", embedding)
    ///     >>> print(f"Created: {result2['created']}")  # False (duplicate)
    #[pyo3(signature = (content, embedding, metadata=None, timestamp=None, source=None, namespace=None))]
    fn add_with_dedup(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<f64>,
        source: Option<&PyDict>,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let engine = self.get_engine()?;
        let ts = timestamp.map(|t| Timestamp::from_unix_secs(t));

        // Parse source from Python dict
        let rust_source = if let Some(src_dict) = source {
            Some(parse_source_from_dict(src_dict)?)
        } else {
            None
        };

        let result = engine
            .add_with_dedup(content, embedding, metadata, ts, rust_source, namespace)
            .map_err(|e| PyValueError::new_err(format!("Failed to add with dedup: {}", e)))?;

        // Convert result to Python dict
        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("id", result.id.to_string())?;
            result_dict.set_item("created", result.created)?;
            if let Some(existing) = result.existing_id {
                result_dict.set_item("existing_id", existing.to_string())?;
            } else {
                result_dict.set_item("existing_id", py.None())?;
            }
            Ok(result_dict.into())
        })
    }

    /// Upsert a memory by logical key
    ///
    /// If key exists: replaces content, embedding, and metadata of existing memory
    /// If key doesn't exist: creates new memory and associates with key
    ///
    /// Args:
    ///     key: Logical key for the memory (e.g., "user_profile:123")
    ///     content: Text content to store
    ///     embedding: Vector embedding (list of floats)
    ///     metadata: Optional metadata dictionary
    ///     timestamp: Optional Unix timestamp (seconds since epoch)
    ///     source: Optional source/provenance tracking dictionary
    ///     namespace: Optional namespace string for multi-user/multi-context isolation
    ///
    /// Returns:
    ///     Dictionary with results:
    ///         - id: Memory ID
    ///         - created: True if new memory created
    ///         - updated: True if existing memory updated
    ///         - previous_content: Previous content if updated (None if created)
    ///
    /// Example:
    ///     >>> embedding = [0.1] * 384
    ///     >>> result1 = memory.upsert("user:123", "Alice likes hiking", embedding)
    ///     >>> print(f"Created: {result1['created']}")  # True
    ///     >>> result2 = memory.upsert("user:123", "Alice likes hiking and photography", embedding)
    ///     >>> print(f"Updated: {result2['updated']}")  # True
    ///     >>> print(f"Previous: {result2['previous_content']}")  # "Alice likes hiking"
    #[pyo3(signature = (key, content, embedding, metadata=None, timestamp=None, source=None, namespace=None))]
    fn upsert(
        &self,
        key: String,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<f64>,
        source: Option<&PyDict>,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let engine = self.get_engine()?;
        let ts = timestamp.map(|t| Timestamp::from_unix_secs(t));

        // Parse source from Python dict
        let rust_source = if let Some(src_dict) = source {
            Some(parse_source_from_dict(src_dict)?)
        } else {
            None
        };

        let result = engine
            .upsert(
                &key,
                content,
                embedding,
                metadata,
                ts,
                rust_source,
                namespace,
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to upsert: {}", e)))?;

        // Convert result to Python dict
        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("id", result.id.to_string())?;
            result_dict.set_item("created", result.created)?;
            result_dict.set_item("updated", result.updated)?;
            if let Some(prev) = result.previous_content {
                result_dict.set_item("previous_content", prev)?;
            } else {
                result_dict.set_item("previous_content", py.None())?;
            }
            Ok(result_dict.into())
        })
    }

    /// Semantic similarity search
    ///
    /// Args:
    ///     query_embedding: Query vector (list of floats)
    ///     top_k: Number of results to return
    ///     namespace: Optional namespace filter. Only returns memories from this namespace
    ///
    /// Returns:
    ///     List of (memory_dict, similarity_score) tuples
    ///
    /// Example:
    ///     >>> query_embedding = [0.1] * 384
    ///     >>> results = memory.search(query_embedding, top_k=10)
    ///     >>> results = memory.search(query_embedding, top_k=10, namespace="user_123")
    ///     >>> # With filters
    ///     >>> filters = [{"field": "type", "op": "eq", "value": "event"}]
    ///     >>> results = memory.search(query_embedding, top_k=10, filters=filters)
    ///     >>> for mem, score in results:
    ///     >>>     print(f"{score:.3f}: {mem['content']}")
    #[pyo3(signature = (query_embedding, top_k, namespace=None, filters=None))]
    fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        namespace: Option<&str>,
        filters: Option<&PyList>,
    ) -> PyResult<Vec<PyObject>> {
        let engine = self.get_engine()?;

        // Parse filters
        let parsed_filters = parse_filters_from_list(filters)?;

        let results = engine
            .search(
                &query_embedding,
                top_k,
                namespace,
                parsed_filters.as_ref().map(|v| v.as_slice()),
            )
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
                    dict.set_item("namespace", memory.get_namespace())?;

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
    ///     namespace: Optional namespace filter. Only returns memories from this namespace
    ///
    /// Returns:
    ///     Tuple of (intent_dict, results_list, profile_context)
    ///     - intent_dict: {"intent": str, "confidence": float}
    ///     - results_list: List of (memory_dict, scores_dict) tuples (real memories only)
    ///     - profile_context: List of profile context strings (summaries or formatted facts)
    ///
    /// Example:
    ///     >>> intent, results, profile_ctx = memory.query("Why was the meeting cancelled?", embedding, 10)
    ///     >>> intent, results, profile_ctx = memory.query("Why?", embedding, 10, namespace="user_123")
    ///     >>> # With filters
    ///     >>> filters = [{"field": "priority", "op": "eq", "value": "high"}]
    ///     >>> intent, results, profile_ctx = memory.query("meetings", embedding, 10, filters=filters)
    ///     >>> print(f"Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
    ///     >>> print(f"Profile context: {len(profile_ctx)} entries")
    ///     >>> for mem, scores in results:
    ///     >>>     print(f"Fused score: {scores['fused_score']:.3f}")
    ///     >>>     print(f"  Content: {mem['content']}")
    #[pyo3(signature = (query_text, query_embedding, limit, namespace=None, filters=None))]
    fn query(
        &self,
        query_text: &str,
        query_embedding: Vec<f32>,
        limit: usize,
        namespace: Option<&str>,
        filters: Option<&PyList>,
    ) -> PyResult<PyObject> {
        let engine = self.get_engine()?;

        // Parse filters
        let parsed_filters = parse_filters_from_list(filters)?;

        let (intent, results, profile_context) = engine
            .query(
                query_text,
                &query_embedding,
                limit,
                namespace,
                parsed_filters.as_ref().map(|v| v.as_slice()),
            )
            .map_err(|e| PyIOError::new_err(format!("Query failed: {}", e)))?;

        Python::with_gil(|py| {
            // Build intent dict
            let intent_dict = PyDict::new(py);
            intent_dict.set_item("intent", format!("{:?}", intent.intent))?;
            intent_dict.set_item("confidence", intent.confidence)?;

            // Build results list (real memories only, no profile facts mixed in)
            let results_list: PyResult<Vec<PyObject>> = results
                .into_iter()
                .map(|(memory, fused_result)| {
                    let mem_dict = PyDict::new(py);
                    mem_dict.set_item("id", memory.id.to_string())?;
                    mem_dict.set_item("content", memory.content.clone())?;
                    mem_dict.set_item("embedding", memory.embedding.clone())?;
                    mem_dict.set_item("metadata", memory.metadata.clone())?;
                    mem_dict.set_item("created_at", memory.created_at.as_unix_secs())?;
                    mem_dict.set_item("namespace", memory.get_namespace())?;

                    // Add source if present
                    if let Ok(Some(source)) = memory.get_source() {
                        let source_dict = source_to_pydict(py, &source)?;
                        mem_dict.set_item("source", source_dict)?;
                    } else {
                        mem_dict.set_item("source", py.None())?;
                    }

                    let scores_dict = PyDict::new(py);
                    scores_dict.set_item("semantic_score", fused_result.semantic_score)?;
                    scores_dict.set_item("bm25_score", fused_result.bm25_score)?;
                    scores_dict.set_item("temporal_score", fused_result.temporal_score)?;
                    scores_dict.set_item("causal_score", fused_result.causal_score)?;
                    scores_dict.set_item("entity_score", fused_result.entity_score)?;
                    scores_dict.set_item("fused_score", fused_result.fused_score)?;

                    let tuple = (mem_dict, scores_dict);
                    Ok(tuple.into_py(py))
                })
                .collect();

            // Profile context as separate list of strings
            let profile_ctx_list: Vec<String> = profile_context;

            let tuple = (intent_dict, results_list?, profile_ctx_list);
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

    /// Reserve capacity in the vector index for future insertions
    ///
    /// This improves performance when adding many memories by avoiding
    /// repeated reallocations. Call this before bulk insertions.
    ///
    /// Args:
    ///     capacity: Number of vectors to reserve space for
    ///
    /// Example:
    ///     >>> memory = Memory("brain.mfdb")
    ///     >>> memory.reserve_capacity(10000)  # Reserve for 10k memories
    ///     >>> # Now add memories efficiently
    fn reserve_capacity(&self, capacity: usize) -> PyResult<()> {
        let engine = self.get_engine()?;
        engine
            .reserve_capacity(capacity)
            .map_err(|e| PyIOError::new_err(format!("Failed to reserve capacity: {}", e)))
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

    /// List all entity profiles in the database
    ///
    /// Entity profiles aggregate facts about entities across all memories,
    /// extracted by the LLM entity extraction pipeline.
    ///
    /// Returns:
    ///     List of entity profile dictionaries with keys:
    ///         - name: Entity name (e.g., "Caroline")
    ///         - entity_type: Type (e.g., "person", "organization")
    ///         - facts: Dict mapping fact_type to list of fact dicts
    ///         - source_memories: List of memory IDs that contributed facts
    ///         - total_facts: Total number of facts in the profile
    ///
    /// Example:
    ///     >>> profiles = memory.list_entity_profiles()
    ///     >>> for p in profiles:
    ///     >>>     print(f"{p['name']} ({p['entity_type']}): {p['total_facts']} facts")
    fn list_entity_profiles(&self) -> PyResult<Vec<PyObject>> {
        let engine = self.get_engine()?;
        let profiles = engine
            .list_entity_profiles()
            .map_err(|e| PyIOError::new_err(format!("Failed to list entity profiles: {}", e)))?;

        Python::with_gil(|py| {
            profiles
                .into_iter()
                .map(|profile| {
                    let dict = PyDict::new(py);
                    dict.set_item("name", &profile.name)?;
                    dict.set_item("entity_type", &profile.entity_type)?;
                    dict.set_item("entity_id", profile.entity_id.to_string())?;
                    dict.set_item("total_facts", profile.total_facts())?;

                    // Convert source_memories to list of strings
                    let source_ids: Vec<String> = profile
                        .source_memories
                        .iter()
                        .map(|id| id.to_string())
                        .collect();
                    dict.set_item("source_memories", source_ids)?;

                    // Convert facts HashMap to Python dict of lists
                    let facts_dict = PyDict::new(py);
                    for (fact_type, facts) in &profile.facts {
                        let fact_list: Vec<PyObject> = facts
                            .iter()
                            .map(|fact| {
                                let f = PyDict::new(py);
                                f.set_item("fact_type", &fact.fact_type).ok();
                                f.set_item("value", &fact.value).ok();
                                f.set_item("confidence", fact.confidence).ok();
                                f.set_item("source_memory", fact.source_memory.to_string()).ok();
                                f.into()
                            })
                            .collect();
                        facts_dict.set_item(fact_type, fact_list)?;
                    }
                    dict.set_item("facts", facts_dict)?;

                    // Include summary if available
                    match &profile.summary {
                        Some(s) => dict.set_item("summary", s)?,
                        None => dict.set_item("summary", py.None())?,
                    }

                    Ok(dict.into())
                })
                .collect()
        })
    }

    /// Get an entity profile by name
    ///
    /// Args:
    ///     name: Entity name (case-insensitive lookup)
    ///
    /// Returns:
    ///     Entity profile dictionary (same format as list_entity_profiles), or None
    ///
    /// Example:
    ///     >>> profile = memory.get_entity_profile("Caroline")
    ///     >>> if profile:
    ///     >>>     for fact_type, facts in profile['facts'].items():
    ///     >>>         for f in facts:
    ///     >>>             print(f"  {fact_type}: {f['value']} (conf={f['confidence']})")
    fn get_entity_profile(&self, name: &str) -> PyResult<Option<PyObject>> {
        let engine = self.get_engine()?;
        let profile = engine
            .get_entity_profile(name)
            .map_err(|e| PyIOError::new_err(format!("Failed to get entity profile: {}", e)))?;

        if let Some(profile) = profile {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("name", &profile.name)?;
                dict.set_item("entity_type", &profile.entity_type)?;
                dict.set_item("entity_id", profile.entity_id.to_string())?;
                dict.set_item("total_facts", profile.total_facts())?;

                let source_ids: Vec<String> = profile
                    .source_memories
                    .iter()
                    .map(|id| id.to_string())
                    .collect();
                dict.set_item("source_memories", source_ids)?;

                let facts_dict = PyDict::new(py);
                for (fact_type, facts) in &profile.facts {
                    let fact_list: Vec<PyObject> = facts
                        .iter()
                        .map(|fact| {
                            let f = PyDict::new(py);
                            f.set_item("fact_type", &fact.fact_type).ok();
                            f.set_item("value", &fact.value).ok();
                            f.set_item("confidence", fact.confidence).ok();
                            f.set_item("source_memory", fact.source_memory.to_string()).ok();
                            f.into()
                        })
                        .collect();
                    facts_dict.set_item(fact_type, fact_list)?;
                }
                dict.set_item("facts", facts_dict)?;

                // Include summary if available
                match &profile.summary {
                    Some(s) => dict.set_item("summary", s)?,
                    None => dict.set_item("summary", py.None())?,
                }

                Ok(Some(dict.into()))
            })
        } else {
            Ok(None)
        }
    }

    /// List all namespaces in the database
    ///
    /// Returns:
    ///     List of namespace strings (excludes default namespace "")
    ///
    /// Example:
    ///     >>> namespaces = memory.list_namespaces()
    ///     >>> print(f"Found {len(namespaces)} namespaces")
    ///     >>> for ns in namespaces:
    ///     >>>     print(f"  - {ns}")
    fn list_namespaces(&self) -> PyResult<Vec<String>> {
        let engine = self.get_engine()?;
        engine
            .list_namespaces()
            .map_err(|e| PyIOError::new_err(format!("Failed to list namespaces: {}", e)))
    }

    /// Count memories in a specific namespace
    ///
    /// Args:
    ///     namespace: Namespace to count (empty string "" for default namespace)
    ///
    /// Returns:
    ///     Number of memories in the namespace
    ///
    /// Example:
    ///     >>> count = memory.count_namespace("user_123")
    ///     >>> print(f"User has {count} memories")
    ///     >>> default_count = memory.count_namespace("")  # Default namespace
    fn count_namespace(&self, namespace: &str) -> PyResult<usize> {
        let engine = self.get_engine()?;
        engine
            .count_namespace(namespace)
            .map_err(|e| PyIOError::new_err(format!("Failed to count namespace: {}", e)))
    }

    /// Delete all memories in a namespace
    ///
    /// Warning: This operation cannot be undone. Use with caution.
    ///
    /// Args:
    ///     namespace: Namespace to delete (empty string "" for default namespace)
    ///
    /// Returns:
    ///     Number of memories deleted
    ///
    /// Example:
    ///     >>> deleted = memory.delete_namespace("old_user")
    ///     >>> print(f"Deleted {deleted} memories from namespace")
    fn delete_namespace(&self, namespace: &str) -> PyResult<usize> {
        let engine = self.get_engine()?;
        engine
            .delete_namespace(namespace)
            .map_err(|e| PyIOError::new_err(format!("Failed to delete namespace: {}", e)))
    }

    /// Enable native LLM entity extraction using Qwen3 models
    ///
    /// This uses native Rust inference with llama-cpp-2 for entity extraction
    /// at ingestion time. Much more accurate than the Python SLM approach.
    ///
    /// Args:
    ///     model_path: Path to the GGUF model file (e.g., Qwen3-4B-Instruct-2507.Q4_K_M.gguf)
    ///     tier: Model tier - "balanced" (4B) or "quality" (8B)
    ///
    /// Returns:
    ///     True if successfully enabled
    ///
    /// Example:
    ///     >>> memory.enable_llm_entity_extraction("models/qwen3-4b/Qwen3-4B-Instruct-2507.Q4_K_M.gguf", "balanced")
    /// Enable native LLM entity extraction using Qwen3 models
    ///
    /// Args:
    ///     model_path: Path to the GGUF model file
    ///     tier: Model tier - "balanced" (4B) or "quality" (8B)
    ///     extraction_passes: Number of extraction passes per document (1-10, default 1).
    ///         Multiple passes capture different facts, producing richer profiles.
    ///         Recommended: 3 for quality. Increases ingestion time linearly.
    ///
    /// Returns:
    ///     True if successfully enabled
    #[cfg(feature = "entity-extraction")]
    #[pyo3(signature = (model_path, tier="balanced", extraction_passes=1))]
    fn enable_llm_entity_extraction(
        &self,
        model_path: &str,
        tier: &str,
        extraction_passes: usize,
    ) -> PyResult<bool> {
        use mnemefusion_core::extraction::ModelTier;

        let model_tier = match tier {
            "balanced" | "4b" => ModelTier::Balanced,
            "quality" | "8b" => ModelTier::Quality,
            _ => return Err(PyValueError::new_err(
                "tier must be 'balanced' (4B) or 'quality' (8B)"
            )),
        };

        if extraction_passes == 0 || extraction_passes > 10 {
            return Err(PyValueError::new_err(
                "extraction_passes must be between 1 and 10"
            ));
        }

        let mut engine_opt = self.engine.borrow_mut();
        if engine_opt.is_none() {
            return Err(PyRuntimeError::new_err("Database is closed"));
        }

        let engine = engine_opt.take().unwrap();
        let new_engine = engine
            .with_llm_entity_extraction_from_path(model_path, model_tier)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to enable LLM extraction: {}", e)))?;

        // Set extraction_passes on the engine's config if > 1
        // The pipeline picks it up from the config wired in open()
        // But since we re-create the engine here, we need to set it directly
        // via the pipeline builder — however the pipeline is internal.
        // Instead, update the config and let the engine wire it.
        // Actually, the pipeline is already built. We need to rebuild with passes.
        // The cleanest approach: accept passes in the config dict at __init__ time.
        // For the API compatibility, we'll store it and it takes effect on next
        // enable_llm call through the pipeline builder.
        //
        // Actually, with_llm_entity_extraction_from_path already creates the pipeline.
        // We need a way to set extraction_passes after that. Let's add a method.

        *engine_opt = Some(new_engine);

        // Set extraction passes on the engine's pipeline
        if extraction_passes > 1 {
            let engine = engine_opt.as_mut().unwrap();
            engine.set_extraction_passes(extraction_passes);
        }

        Ok(true)
    }

    /// Run entity extraction on text without adding to the database.
    ///
    /// Useful for testing extraction quality or comparing model outputs.
    /// Requires enable_llm_entity_extraction() to have been called first.
    ///
    /// Args:
    ///     content: The text to extract entities from
    ///     speaker: Optional speaker name for first-person attribution
    ///
    /// Returns:
    ///     Dict with keys: entities, entity_facts, topics, importance, records, relationships
    ///
    /// Example:
    ///     >>> result = memory.extract_text("Alice works at Google", speaker="Alice")
    ///     >>> print(result["entity_facts"])
    #[cfg(feature = "entity-extraction")]
    #[pyo3(signature = (content, speaker=None))]
    fn extract_text(&self, py: Python, content: &str, speaker: Option<&str>) -> PyResult<PyObject> {
        let engine = self.get_engine()?;
        let result = engine
            .extract_text(content, speaker)
            .map_err(|e| PyRuntimeError::new_err(format!("Extraction failed: {}", e)))?;

        // Convert ExtractionResult to Python dict
        let dict = PyDict::new(py);

        // Entities
        let ent_list = PyList::empty(py);
        for e in &result.entities {
            let d = PyDict::new(py);
            d.set_item("name", &e.name)?;
            d.set_item("type", &e.entity_type)?;
            ent_list.append(d)?;
        }
        dict.set_item("entities", ent_list)?;

        // Entity facts
        let fact_list = PyList::empty(py);
        for f in &result.entity_facts {
            let d = PyDict::new(py);
            d.set_item("entity", &f.entity)?;
            d.set_item("fact_type", &f.fact_type)?;
            d.set_item("value", &f.value)?;
            d.set_item("confidence", f.confidence)?;
            fact_list.append(d)?;
        }
        dict.set_item("entity_facts", fact_list)?;

        // Topics
        let topic_list = PyList::new(py, &result.topics);
        dict.set_item("topics", topic_list)?;

        // Importance
        dict.set_item("importance", result.importance)?;

        // Records
        let rec_list = PyList::empty(py);
        for r in &result.records {
            let d = PyDict::new(py);
            d.set_item("record_type", &r.record_type)?;
            d.set_item("summary", &r.summary)?;
            if let Some(ref date) = r.event_date {
                d.set_item("event_date", date)?;
            }
            let ents = PyList::new(py, &r.entities);
            d.set_item("entities", ents)?;
            rec_list.append(d)?;
        }
        dict.set_item("records", rec_list)?;

        // Relationships
        let rel_list = PyList::empty(py);
        for r in &result.relationships {
            let d = PyDict::new(py);
            d.set_item("from_entity", &r.from_entity)?;
            d.set_item("to_entity", &r.to_entity)?;
            d.set_item("relation_type", &r.relation_type)?;
            d.set_item("confidence", r.confidence)?;
            rel_list.append(d)?;
        }
        dict.set_item("relationships", rel_list)?;

        Ok(dict.into())
    }

    /// Set an embedding function for computing fact embeddings at ingestion time
    ///
    /// The embedding function is called for each entity fact extracted during ingestion.
    /// Fact embeddings enable semantic matching in ProfileSearch (cosine similarity
    /// instead of word overlap). Facts without embeddings fall back to word overlap.
    ///
    /// Args:
    ///     func: A callable that takes a string and returns a list of floats (embedding vector).
    ///           Typically a sentence-transformers model.encode() wrapper.
    ///
    /// Example:
    ///     >>> from sentence_transformers import SentenceTransformer
    ///     >>> model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    ///     >>> memory.set_embedding_fn(lambda text: model.encode(text).tolist())
    fn set_embedding_fn(&self, func: PyObject) -> PyResult<()> {
        let func = Arc::new(func);
        let embed_fn: EmbeddingFn = Arc::new(move |text: &str| {
            Python::with_gil(|py| {
                let result = func
                    .call1(py, (text,))
                    .expect("embedding_fn call failed");
                result
                    .extract::<Vec<f32>>(py)
                    .expect("embedding_fn must return List[float]")
            })
        });

        let mut engine_opt = self.engine.borrow_mut();
        let engine = engine_opt
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        engine.set_embedding_fn(embed_fn);
        Ok(())
    }

    /// Precompute fact embeddings for all entity profiles.
    ///
    /// Call this after set_embedding_fn() to backfill fact embeddings
    /// for profiles created before embedding support was added.
    ///
    /// Returns: Number of fact embeddings computed.
    fn precompute_fact_embeddings(&self) -> PyResult<usize> {
        let engine = self.get_engine()?;
        engine
            .precompute_fact_embeddings()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to precompute: {}", e)))
    }

    /// Consolidate entity profiles by removing noise and deduplicating facts.
    ///
    /// Performs:
    /// 1. Remove null-indicator values ("none", "N/A", etc.)
    /// 2. Remove overly verbose values (>100 chars)
    /// 3. Semantic dedup within same fact_type (cosine similarity > 0.85)
    /// 4. Delete garbage entity profiles (non-person with ≤2 facts)
    ///
    /// Returns: Tuple of (facts_removed, profiles_deleted)
    fn consolidate_profiles(&self) -> PyResult<(usize, usize)> {
        let engine = self.get_engine()?;
        engine
            .consolidate_profiles()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to consolidate: {}", e)))
    }

    /// Apply an externally-produced extraction result to a memory's entity profiles.
    ///
    /// Enables API-based extraction backends (e.g., NScale cloud inference) to inject
    /// entity profiles without requiring a local LLM. The extraction dict must match
    /// the same JSON schema as the local Qwen3 extractor.
    ///
    /// Args:
    ///     memory_id: The memory ID (string) to associate the extraction with
    ///     extraction: Dict with keys: entities, entity_facts, topics, importance
    ///
    /// Example:
    ///     >>> extraction = {
    ///     ...     "entities": [{"name": "Alice", "type": "person"}],
    ///     ...     "entity_facts": [{"entity": "Alice", "fact_type": "hobby", "value": "hiking", "confidence": 0.9}],
    ///     ...     "topics": ["outdoors"],
    ///     ...     "importance": 0.8,
    ///     ... }
    ///     >>> memory.apply_extraction("mem-abc123", extraction)
    #[cfg(feature = "entity-extraction")]
    fn apply_extraction(&self, memory_id: &str, extraction: &PyDict) -> PyResult<()> {
        use mnemefusion_core::extraction::{
            ExtractedEntity, ExtractedFact, ExtractedRelationship, ExtractionResult, TypedRecord,
        };

        let engine = self.get_engine()?;
        let mid = MemoryId::parse(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        // Parse entities
        let entities: Vec<ExtractedEntity> = if let Some(ents) = extraction.get_item("entities")? {
            let ent_list: &PyList = ents.downcast().map_err(|_| {
                PyValueError::new_err("'entities' must be a list")
            })?;
            let mut result = Vec::new();
            for item in ent_list.iter() {
                let d: &PyDict = item.downcast().map_err(|_| {
                    PyValueError::new_err("Each entity must be a dict")
                })?;
                let name: String = d
                    .get_item("name")?
                    .ok_or_else(|| PyValueError::new_err("Entity missing 'name'"))?
                    .extract()?;
                let entity_type: String = d
                    .get_item("type")?
                    .ok_or_else(|| PyValueError::new_err("Entity missing 'type'"))?
                    .extract()?;
                result.push(ExtractedEntity { name, entity_type });
            }
            result
        } else {
            Vec::new()
        };

        // Parse entity_facts
        let entity_facts: Vec<ExtractedFact> =
            if let Some(facts) = extraction.get_item("entity_facts")? {
                let fact_list: &PyList = facts.downcast().map_err(|_| {
                    PyValueError::new_err("'entity_facts' must be a list")
                })?;
                let mut result = Vec::new();
                for item in fact_list.iter() {
                    let d: &PyDict = item.downcast().map_err(|_| {
                        PyValueError::new_err("Each entity_fact must be a dict")
                    })?;
                    let entity: String = d
                        .get_item("entity")?
                        .ok_or_else(|| PyValueError::new_err("Fact missing 'entity'"))?
                        .extract()?;
                    let fact_type: String = d
                        .get_item("fact_type")?
                        .ok_or_else(|| PyValueError::new_err("Fact missing 'fact_type'"))?
                        .extract()?;
                    let value: String = d
                        .get_item("value")?
                        .ok_or_else(|| PyValueError::new_err("Fact missing 'value'"))?
                        .extract()?;
                    let confidence: f32 = d
                        .get_item("confidence")?
                        .map(|v| v.extract().unwrap_or(0.9))
                        .unwrap_or(0.9);
                    result.push(ExtractedFact {
                        entity,
                        fact_type,
                        value,
                        confidence,
                    });
                }
                result
            } else {
                Vec::new()
            };

        // Parse topics
        let topics: Vec<String> = if let Some(t) = extraction.get_item("topics")? {
            t.extract().unwrap_or_default()
        } else {
            Vec::new()
        };

        // Parse importance
        let importance: f32 = if let Some(imp) = extraction.get_item("importance")? {
            imp.extract().unwrap_or(0.5)
        } else {
            0.5
        };

        // Parse records (typed sub-records)
        let records: Vec<TypedRecord> = if let Some(recs) = extraction.get_item("records")? {
            let rec_list: &PyList = recs.downcast().map_err(|_| {
                PyValueError::new_err("'records' must be a list")
            })?;
            let mut result = Vec::new();
            for item in rec_list.iter() {
                let d: &PyDict = item.downcast().map_err(|_| {
                    PyValueError::new_err("Each record must be a dict")
                })?;
                let record_type: String = d
                    .get_item("record_type")?
                    .ok_or_else(|| PyValueError::new_err("Record missing 'record_type'"))?
                    .extract()?;
                let summary: String = d
                    .get_item("summary")?
                    .ok_or_else(|| PyValueError::new_err("Record missing 'summary'"))?
                    .extract()?;
                let event_date: Option<String> = d
                    .get_item("event_date")?
                    .and_then(|v| v.extract().ok());
                let record_entities: Vec<String> = d
                    .get_item("entities")?
                    .map(|v| v.extract().unwrap_or_default())
                    .unwrap_or_default();
                result.push(TypedRecord {
                    record_type,
                    summary,
                    event_date,
                    entities: record_entities,
                });
            }
            result
        } else {
            Vec::new()
        };

        // Parse relationships
        let relationships: Vec<ExtractedRelationship> =
            if let Some(rels) = extraction.get_item("relationships")? {
                let rel_list: &PyList = rels.downcast().map_err(|_| {
                    PyValueError::new_err("'relationships' must be a list")
                })?;
                let mut result = Vec::new();
                for item in rel_list.iter() {
                    let d: &PyDict = item.downcast().map_err(|_| {
                        PyValueError::new_err("Each relationship must be a dict")
                    })?;
                    let from_entity: String = d
                        .get_item("from_entity")?
                        .ok_or_else(|| PyValueError::new_err("Relationship missing 'from_entity'"))?
                        .extract()?;
                    let to_entity: String = d
                        .get_item("to_entity")?
                        .ok_or_else(|| PyValueError::new_err("Relationship missing 'to_entity'"))?
                        .extract()?;
                    let relation_type: String = d
                        .get_item("relation_type")?
                        .ok_or_else(|| PyValueError::new_err("Relationship missing 'relation_type'"))?
                        .extract()?;
                    let confidence: f32 = d
                        .get_item("confidence")?
                        .map(|v| v.extract().unwrap_or(0.9))
                        .unwrap_or(0.9);
                    result.push(ExtractedRelationship {
                        from_entity,
                        to_entity,
                        relation_type,
                        confidence,
                    });
                }
                result
            } else {
                Vec::new()
            };

        let extraction_result = ExtractionResult {
            entities,
            entity_facts,
            topics,
            importance,
            records,
            relationships,
        };

        engine
            .apply_extraction(&mid, &extraction_result)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to apply extraction: {}", e)))
    }

    /// Generate summaries for all entity profiles.
    ///
    /// For each profile with facts, generates a dense summary paragraph that
    /// condenses the profile's facts into one text block. When present, query()
    /// injects summaries as single context items instead of N individual facts.
    ///
    /// Returns: Number of profiles summarized
    ///
    /// Example:
    ///     >>> count = memory.summarize_profiles()
    ///     >>> print(f"Summarized {count} profiles")
    fn summarize_profiles(&self) -> PyResult<usize> {
        let engine = self.get_engine()?;
        engine
            .summarize_profiles()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to summarize: {}", e)))
    }

    /// Get the N most recent memories
    ///
    /// Args:
    ///     n: Number of recent memories to retrieve
    ///     namespace: Optional namespace filter
    ///
    /// Returns:
    ///     List of (memory_dict, timestamp) tuples, sorted newest first
    ///
    /// Example:
    ///     >>> recent = memory.get_recent(10)
    ///     >>> for mem, ts in recent:
    ///     >>>     print(f"{ts}: {mem['content']}")
    #[pyo3(signature = (n, namespace=None))]
    fn get_recent(&self, n: usize, namespace: Option<&str>) -> PyResult<Vec<PyObject>> {
        let engine = self.get_engine()?;
        let results = engine
            .get_recent(n, namespace)
            .map_err(|e| PyIOError::new_err(format!("Failed to get recent: {}", e)))?;

        Python::with_gil(|py| {
            results
                .into_iter()
                .map(|(memory, timestamp)| {
                    let dict = PyDict::new(py);
                    dict.set_item("id", memory.id.to_string())?;
                    dict.set_item("content", memory.content.clone())?;
                    dict.set_item("embedding", memory.embedding.clone())?;
                    dict.set_item("metadata", memory.metadata.clone())?;
                    dict.set_item("created_at", memory.created_at.as_unix_secs())?;
                    dict.set_item("namespace", memory.get_namespace())?;

                    if let Ok(Some(source)) = memory.get_source() {
                        let source_dict = source_to_pydict(py, &source)?;
                        dict.set_item("source", source_dict)?;
                    } else {
                        dict.set_item("source", py.None())?;
                    }

                    let tuple = (dict, timestamp.as_unix_secs());
                    Ok(tuple.into_py(py))
                })
                .collect()
        })
    }

    /// Get memories within a time range
    ///
    /// Args:
    ///     start: Start timestamp (Unix seconds, float)
    ///     end: End timestamp (Unix seconds, float)
    ///     limit: Maximum number of results
    ///     namespace: Optional namespace filter
    ///
    /// Returns:
    ///     List of (memory_dict, timestamp) tuples, sorted newest first
    ///
    /// Example:
    ///     >>> import time
    ///     >>> end = time.time()
    ///     >>> start = end - 86400  # Last 24 hours
    ///     >>> results = memory.get_range(start, end, 100)
    #[pyo3(signature = (start, end, limit, namespace=None))]
    fn get_range(
        &self,
        start: f64,
        end: f64,
        limit: usize,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let engine = self.get_engine()?;
        let results = engine
            .get_range(
                Timestamp::from_unix_secs(start),
                Timestamp::from_unix_secs(end),
                limit,
                namespace,
            )
            .map_err(|e| PyIOError::new_err(format!("Failed to get range: {}", e)))?;

        Python::with_gil(|py| {
            results
                .into_iter()
                .map(|(memory, timestamp)| {
                    let dict = PyDict::new(py);
                    dict.set_item("id", memory.id.to_string())?;
                    dict.set_item("content", memory.content.clone())?;
                    dict.set_item("embedding", memory.embedding.clone())?;
                    dict.set_item("metadata", memory.metadata.clone())?;
                    dict.set_item("created_at", memory.created_at.as_unix_secs())?;
                    dict.set_item("namespace", memory.get_namespace())?;

                    if let Ok(Some(source)) = memory.get_source() {
                        let source_dict = source_to_pydict(py, &source)?;
                        dict.set_item("source", source_dict)?;
                    } else {
                        dict.set_item("source", py.None())?;
                    }

                    let tuple = (dict, timestamp.as_unix_secs());
                    Ok(tuple.into_py(py))
                })
                .collect()
        })
    }

    /// Get all memories mentioning an entity
    ///
    /// Args:
    ///     entity_name: Name of the entity (case-insensitive)
    ///
    /// Returns:
    ///     List of memory dictionaries
    ///
    /// Example:
    ///     >>> memories = memory.get_entity_memories("Alice")
    ///     >>> for mem in memories:
    ///     >>>     print(mem['content'])
    fn get_entity_memories(&self, entity_name: &str) -> PyResult<Vec<PyObject>> {
        let engine = self.get_engine()?;
        let memories = engine
            .get_entity_memories(entity_name)
            .map_err(|e| PyIOError::new_err(format!("Failed to get entity memories: {}", e)))?;

        Python::with_gil(|py| {
            memories
                .into_iter()
                .map(|memory| {
                    let dict = PyDict::new(py);
                    dict.set_item("id", memory.id.to_string())?;
                    dict.set_item("content", memory.content.clone())?;
                    dict.set_item("embedding", memory.embedding.clone())?;
                    dict.set_item("metadata", memory.metadata.clone())?;
                    dict.set_item("created_at", memory.created_at.as_unix_secs())?;
                    dict.set_item("namespace", memory.get_namespace())?;

                    if let Ok(Some(source)) = memory.get_source() {
                        let source_dict = source_to_pydict(py, &source)?;
                        dict.set_item("source", source_dict)?;
                    } else {
                        dict.set_item("source", py.None())?;
                    }

                    Ok(dict.into())
                })
                .collect()
        })
    }

    /// Get all entities mentioned in a specific memory
    ///
    /// Args:
    ///     memory_id: Memory ID string
    ///
    /// Returns:
    ///     List of entity dictionaries
    ///
    /// Example:
    ///     >>> entities = memory.get_memory_entities(memory_id)
    ///     >>> for e in entities:
    ///     >>>     print(f"{e['name']} ({e['mention_count']} mentions)")
    fn get_memory_entities(&self, memory_id: &str) -> PyResult<Vec<PyObject>> {
        let engine = self.get_engine()?;
        let id = MemoryId::parse(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;

        let entities = engine
            .get_memory_entities(&id)
            .map_err(|e| PyIOError::new_err(format!("Failed to get memory entities: {}", e)))?;

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

    /// Count entity profiles in the database
    ///
    /// Returns:
    ///     Number of entity profiles
    ///
    /// Example:
    ///     >>> count = memory.count_entity_profiles()
    ///     >>> print(f"Total profiles: {count}")
    fn count_entity_profiles(&self) -> PyResult<usize> {
        let engine = self.get_engine()?;
        engine
            .count_entity_profiles()
            .map_err(|e| PyIOError::new_err(format!("Failed to count entity profiles: {}", e)))
    }

    /// List all memory IDs in the database
    ///
    /// Warning: Loads all IDs into memory. Use with caution on large databases.
    ///
    /// Returns:
    ///     List of memory ID strings
    ///
    /// Example:
    ///     >>> ids = memory.list_ids()
    ///     >>> print(f"Total memories: {len(ids)}")
    fn list_ids(&self) -> PyResult<Vec<String>> {
        let engine = self.get_engine()?;
        let ids = engine
            .list_ids()
            .map_err(|e| PyIOError::new_err(format!("Failed to list IDs: {}", e)))?;
        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    /// Update the embedding vector for an existing memory.
    ///
    /// Updates both the stored memory record and the HNSW vector index.
    /// Content, metadata, and all other fields are preserved.
    ///
    /// Args:
    ///     memory_id: Memory ID string
    ///     embedding: New embedding vector (must match configured dimension)
    ///
    /// Example:
    ///     >>> memory.update_embedding(memory_id, new_embedding)
    fn update_embedding(&self, memory_id: &str, embedding: Vec<f32>) -> PyResult<()> {
        let engine = self.get_engine()?;
        let id = MemoryId::parse(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid memory ID: {}", e)))?;
        engine
            .update_embedding(&id, embedding)
            .map_err(|e| PyIOError::new_err(format!("Failed to update embedding: {}", e)))
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
    fn __exit__(
        &self,
        _exc_type: PyObject,
        _exc_value: PyObject,
        _traceback: PyObject,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Build contextual text for embedding by prepending speaker metadata.
///
/// Call this on content before computing embeddings to make them speaker-aware.
/// The raw content should still be passed to add() for storage.
///
/// Example:
///     >>> ctx_text = mnemefusion.contextualize_for_embedding("I love hiking", {"speaker": "Melanie"})
///     >>> # ctx_text = "Melanie: I love hiking"
///     >>> embedding = model.encode(ctx_text)
///     >>> memory.add("I love hiking", embedding, {"speaker": "Melanie"})
#[pyfunction]
fn contextualize_for_embedding(content: &str, metadata: HashMap<String, String>) -> String {
    mnemefusion_core::contextualize_for_embedding(content, &metadata)
}

/// MnemeFusion Python module
#[pymodule]
fn mnemefusion(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMemory>()?;
    m.add_function(wrap_pyfunction!(contextualize_for_embedding, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
