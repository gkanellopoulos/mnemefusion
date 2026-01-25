//! Graph persistence to storage.
//!
//! Handles serialization and deserialization of causal and entity graph edges.

use crate::{
    graph::causal::{CausalEdge, GraphManager},
    graph::entity::EntityNode,
    storage::StorageEngine,
    types::{EntityId, MemoryId},
    Result,
};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Serializable representation of a causal edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CausalEdgeData {
    source: String, // MemoryId as string
    target: String, // MemoryId as string
    confidence: f32,
    evidence: String,
}

/// Serializable representation of an entity graph edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntityEdgeData {
    memory_id: String, // MemoryId as string
    entity_id: String, // EntityId as string
}

/// Save the causal and entity graphs to storage.
///
/// Serializes all edges in both graphs and stores them in the storage tables.
pub fn save_graph(graph: &GraphManager, storage: &Arc<StorageEngine>) -> Result<()> {
    // Save causal graph
    let causal_edges = graph.edges();

    // Convert to serializable format
    let causal_edge_data: Vec<CausalEdgeData> = causal_edges
        .into_iter()
        .map(|(source, target, edge)| CausalEdgeData {
            source: source.to_string(),
            target: target.to_string(),
            confidence: edge.confidence,
            evidence: edge.evidence,
        })
        .collect();

    // Serialize all causal edges as a single JSON array
    let causal_json = serde_json::to_vec(&causal_edge_data)
        .map_err(|e| crate::Error::Serialization(e.to_string()))?;

    // Store in the causal graph table
    storage.store_causal_graph(&causal_json)?;

    // Save entity graph
    let entity_edges = get_entity_edges(graph);

    // Convert to serializable format
    let entity_edge_data: Vec<EntityEdgeData> = entity_edges
        .into_iter()
        .map(|(memory_id, entity_id)| EntityEdgeData {
            memory_id: memory_id.to_string(),
            entity_id: entity_id.to_string(),
        })
        .collect();

    // Serialize entity edges
    let entity_json = serde_json::to_vec(&entity_edge_data)
        .map_err(|e| crate::Error::Serialization(e.to_string()))?;

    // Store in metadata table
    storage.store_entity_graph(&entity_json)?;

    Ok(())
}

/// Helper function to extract entity edges from graph manager
fn get_entity_edges(graph: &GraphManager) -> Vec<(MemoryId, EntityId)> {
    let entity_graph = graph.entity_graph();
    let mut edges = Vec::new();

    // Iterate through all edges in the entity graph
    for edge_ref in entity_graph.graph.edge_references() {
        let source_idx = edge_ref.source();
        let target_idx = edge_ref.target();

        if let (Some(EntityNode::Memory(memory_id)), Some(EntityNode::Entity(entity_id))) = (
            entity_graph.graph.node_weight(source_idx),
            entity_graph.graph.node_weight(target_idx),
        ) {
            edges.push((memory_id.clone(), entity_id.clone()));
        }
    }

    edges
}

/// Load the causal and entity graphs from storage.
///
/// Deserializes edges from storage tables and loads them into the graph.
pub fn load_graph(graph: &mut GraphManager, storage: &Arc<StorageEngine>) -> Result<()> {
    // Load causal graph
    if let Some(causal_json) = storage.load_causal_graph()? {
        // Deserialize
        let edge_data: Vec<CausalEdgeData> = serde_json::from_slice(&causal_json)
            .map_err(|e| crate::Error::Deserialization(e.to_string()))?;

        // Convert back to graph format
        let edges: Vec<(MemoryId, MemoryId, CausalEdge)> = edge_data
            .into_iter()
            .map(|data| {
                let source = MemoryId::parse(&data.source).map_err(|e| {
                    crate::Error::Deserialization(format!("Invalid source ID: {}", e))
                })?;
                let target = MemoryId::parse(&data.target).map_err(|e| {
                    crate::Error::Deserialization(format!("Invalid target ID: {}", e))
                })?;
                let edge = CausalEdge {
                    confidence: data.confidence,
                    evidence: data.evidence,
                };
                Ok((source, target, edge))
            })
            .collect::<Result<Vec<_>>>()?;

        // Load into causal graph
        graph.load_edges(edges)?;
    }

    // Load entity graph
    if let Some(entity_json) = storage.load_entity_graph()? {
        // Deserialize
        let edge_data: Vec<EntityEdgeData> = serde_json::from_slice(&entity_json)
            .map_err(|e| crate::Error::Deserialization(e.to_string()))?;

        // Load edges into entity graph
        for edge in edge_data {
            let memory_id = MemoryId::parse(&edge.memory_id)
                .map_err(|e| crate::Error::Deserialization(format!("Invalid memory ID: {}", e)))?;
            let entity_id = EntityId::parse(&edge.entity_id)
                .map_err(|e| crate::Error::Deserialization(format!("Invalid entity ID: {}", e)))?;

            graph.link_memory_to_entity(&memory_id, &entity_id);
        }
    }

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
        let graph = GraphManager::new();

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
        graph
            .add_causal_link(&m1, &m2, 0.9, "edge1".to_string())
            .unwrap();
        graph
            .add_causal_link(&m2, &m3, 0.8, "edge2".to_string())
            .unwrap();

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
        graph
            .add_causal_link(&m1, &m2, 0.9, "edge1".to_string())
            .unwrap();
        save_graph(&graph, &storage).unwrap();

        // Modify and save again
        let m3 = make_memory_id(3);
        graph
            .add_causal_link(&m2, &m3, 0.8, "edge2".to_string())
            .unwrap();
        save_graph(&graph, &storage).unwrap();

        // Load should get the latest version
        let mut graph2 = GraphManager::new();
        load_graph(&mut graph2, &storage).unwrap();

        assert_eq!(graph2.node_count(), 3);
        assert_eq!(graph2.edge_count(), 2);
    }
}
