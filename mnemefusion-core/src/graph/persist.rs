//! Graph persistence to storage.
//!
//! Handles serialization and deserialization of causal graph edges.

use crate::{
    graph::causal::{CausalEdge, GraphManager},
    storage::StorageEngine,
    types::MemoryId,
    Result,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Serializable representation of a causal edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CausalEdgeData {
    source: String,  // MemoryId as string
    target: String,  // MemoryId as string
    confidence: f32,
    evidence: String,
}

/// Save the causal graph to storage.
///
/// Serializes all edges in the graph and stores them in the CAUSAL_GRAPH table.
pub fn save_graph(graph: &GraphManager, storage: &Arc<StorageEngine>) -> Result<()> {
    let edges = graph.edges();

    // Convert to serializable format
    let edge_data: Vec<CausalEdgeData> = edges
        .into_iter()
        .map(|(source, target, edge)| CausalEdgeData {
            source: source.to_string(),
            target: target.to_string(),
            confidence: edge.confidence,
            evidence: edge.evidence,
        })
        .collect();

    // Serialize all edges as a single JSON array
    let json = serde_json::to_vec(&edge_data)
        .map_err(|e| crate::Error::Serialization(e.to_string()))?;

    // Store in the causal graph table
    storage.store_causal_graph(&json)?;

    Ok(())
}

/// Load the causal graph from storage.
///
/// Deserializes edges from the CAUSAL_GRAPH table and loads them into the graph.
pub fn load_graph(graph: &mut GraphManager, storage: &Arc<StorageEngine>) -> Result<()> {
    // Load from storage
    let json = match storage.load_causal_graph()? {
        Some(data) => data,
        None => return Ok(()), // No graph data yet
    };

    // Deserialize
    let edge_data: Vec<CausalEdgeData> = serde_json::from_slice(&json)
        .map_err(|e| crate::Error::Deserialization(e.to_string()))?;

    // Convert back to graph format
    let edges: Vec<(MemoryId, MemoryId, CausalEdge)> = edge_data
        .into_iter()
        .map(|data| {
            let source = MemoryId::parse(&data.source)
                .map_err(|e| crate::Error::Deserialization(format!("Invalid source ID: {}", e)))?;
            let target = MemoryId::parse(&data.target)
                .map_err(|e| crate::Error::Deserialization(format!("Invalid target ID: {}", e)))?;
            let edge = CausalEdge {
                confidence: data.confidence,
                evidence: data.evidence,
            };
            Ok((source, target, edge))
        })
        .collect::<Result<Vec<_>>>()?;

    // Load into graph
    graph.load_edges(edges)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryId;
    use tempfile::tempdir;

    fn make_memory_id(n: u128) -> MemoryId {
        MemoryId::from_u128(n)
    }

    #[test]
    fn test_save_and_load_empty_graph() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let storage = Arc::new(StorageEngine::open(&path).unwrap());
        let mut graph = GraphManager::new();

        // Save empty graph
        save_graph(&graph, &storage).unwrap();

        // Load into new graph
        let mut graph2 = GraphManager::new();
        load_graph(&mut graph2, &storage).unwrap();

        assert_eq!(graph2.node_count(), 0);
        assert_eq!(graph2.edge_count(), 0);
    }

    #[test]
    fn test_save_and_load_simple_graph() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let storage = Arc::new(StorageEngine::open(&path).unwrap());
        let mut graph = GraphManager::new();

        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);
        let m3 = make_memory_id(3);

        // Build graph
        graph.add_causal_link(&m1, &m2, 0.9, "edge1".to_string()).unwrap();
        graph.add_causal_link(&m2, &m3, 0.8, "edge2".to_string()).unwrap();

        // Save
        save_graph(&graph, &storage).unwrap();

        // Load into new graph
        let mut graph2 = GraphManager::new();
        load_graph(&mut graph2, &storage).unwrap();

        assert_eq!(graph2.node_count(), 3);
        assert_eq!(graph2.edge_count(), 2);

        // Verify edges work
        let result = graph2.get_effects(&m1, 2).unwrap();
        assert_eq!(result.paths.len(), 2); // m1→m2 and m1→m2→m3
    }

    #[test]
    fn test_save_overwrite() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let storage = Arc::new(StorageEngine::open(&path).unwrap());
        let mut graph = GraphManager::new();

        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);

        // Save first graph
        graph.add_causal_link(&m1, &m2, 0.9, "edge1".to_string()).unwrap();
        save_graph(&graph, &storage).unwrap();

        // Modify and save again
        let m3 = make_memory_id(3);
        graph.add_causal_link(&m2, &m3, 0.8, "edge2".to_string()).unwrap();
        save_graph(&graph, &storage).unwrap();

        // Load should get the latest version
        let mut graph2 = GraphManager::new();
        load_graph(&mut graph2, &storage).unwrap();

        assert_eq!(graph2.node_count(), 3);
        assert_eq!(graph2.edge_count(), 2);
    }
}
