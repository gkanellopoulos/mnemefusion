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

/// Serializable representation of an entity-to-entity relationship edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RelationshipEdgeData {
    from_entity_id: String,
    to_entity_id: String,
    relation_type: String,
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

    // Save entity-to-entity relationship edges
    let relationship_edges = get_relationship_edges(graph);
    let relationship_edge_data: Vec<RelationshipEdgeData> = relationship_edges
        .into_iter()
        .map(|(from_id, to_id, relation)| RelationshipEdgeData {
            from_entity_id: from_id.to_string(),
            to_entity_id: to_id.to_string(),
            relation_type: relation,
        })
        .collect();

    let relationship_json = serde_json::to_vec(&relationship_edge_data)
        .map_err(|e| crate::Error::Serialization(e.to_string()))?;

    storage.store_relationship_graph(&relationship_json)?;

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

/// Helper function to extract entity-to-entity relationship edges from graph manager
fn get_relationship_edges(graph: &GraphManager) -> Vec<(EntityId, EntityId, String)> {
    let entity_graph = graph.entity_graph();
    let mut edges = Vec::new();

    for edge_ref in entity_graph.graph.edge_references() {
        let source_idx = edge_ref.source();
        let target_idx = edge_ref.target();

        // Only Entity→Entity edges with non-"mentions" relationship
        if let (Some(EntityNode::Entity(from_id)), Some(EntityNode::Entity(to_id))) = (
            entity_graph.graph.node_weight(source_idx),
            entity_graph.graph.node_weight(target_idx),
        ) {
            if edge_ref.weight().relationship != "mentions" {
                edges.push((
                    from_id.clone(),
                    to_id.clone(),
                    edge_ref.weight().relationship.clone(),
                ));
            }
        }
    }

    edges
}

/// Parse a relationship fact value to extract (relation_type, target_entity_name).
///
/// Handles multiple formats produced by different extraction models:
/// - `"friend of Mel"` → ("friend", "Mel")
/// - `"friendship with Caroline"` → ("friendship", "Caroline")
/// - `"Melanie (colleague or friend)"` → ("colleague or friend", "Melanie")
///
/// For the parenthetical format, `known_names` is used to verify the entity name.
/// Returns None if the value doesn't match any known pattern.
fn parse_relationship_target<'a>(
    value: &'a str,
    known_names: &[String],
) -> Option<(&'a str, &'a str)> {
    // Pattern 1: "{relation_type} of {entity_name}"
    if let Some(pos) = value.find(" of ") {
        return Some((&value[..pos], &value[pos + 4..]));
    }

    // Pattern 2: "{relation_type} with {entity_name}"
    if let Some(pos) = value.find(" with ") {
        return Some((&value[..pos], &value[pos + 6..]));
    }

    // Pattern 3: "{entity_name} ({relation_type})" — Phi-4 parenthetical format
    if value.contains('(') && value.ends_with(')') {
        if let Some(paren_pos) = value.find('(') {
            let entity_candidate = value[..paren_pos].trim();
            let relation = &value[paren_pos + 1..value.len() - 1];

            if entity_candidate.is_empty() || relation.is_empty() {
                return None;
            }

            // Verify entity_candidate matches a known profile name
            let entity_lower = entity_candidate.to_lowercase();
            for name in known_names {
                if name.to_lowercase() == entity_lower {
                    return Some((relation, entity_candidate));
                }
            }
        }
    }

    None
}

/// Repair entity-to-entity relationship edges from profile facts.
///
/// Existing databases have relationship facts stored in entity profiles
/// (by `store_relationships()`) but lost their Entity→Entity graph edges
/// due to the persistence bug (edges were silently dropped on save/load).
///
/// This function reconstructs Entity→Entity edges from profile data.
/// Runs once at DB open when the "relationship_graph" key is absent (one-time migration).
pub fn repair_relationship_edges(
    graph: &mut GraphManager,
    storage: &Arc<StorageEngine>,
) -> Result<()> {
    // Only run if relationship_graph key is absent (one-time migration)
    if storage.has_relationship_graph()? {
        return Ok(());
    }

    let profiles = storage.list_entity_profiles()?;
    let known_names: Vec<String> = profiles.iter().map(|p| p.name.clone()).collect();
    let mut edges_added = 0;

    for profile in &profiles {
        // Find facts with fact_type == "relationship"
        if let Some(rel_facts) = profile.facts.get("relationship") {
            for fact in rel_facts {
                if let Some((relation_type, target_name)) =
                    parse_relationship_target(&fact.value, &known_names)
                {
                    // Find the target entity's profile to get its entity_id
                    if let Ok(Some(target_profile)) = storage.get_entity_profile(target_name) {
                        graph.link_entity_to_entity(
                            &profile.entity_id,
                            &target_profile.entity_id,
                            relation_type,
                        );
                        edges_added += 1;
                    }
                }
            }
        }
    }

    if edges_added > 0 {
        tracing::info!(
            "Repaired {} entity-to-entity relationship edges from profile facts",
            edges_added
        );
        // Save the repaired graph so this doesn't run again
        save_graph(graph, storage)?;
    }

    Ok(())
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

    // Load entity-to-entity relationship edges
    if let Some(relationship_json) = storage.load_relationship_graph()? {
        let edge_data: Vec<RelationshipEdgeData> = serde_json::from_slice(&relationship_json)
            .map_err(|e| crate::Error::Deserialization(e.to_string()))?;

        for edge in edge_data {
            let from_id = EntityId::parse(&edge.from_entity_id)
                .map_err(|e| crate::Error::Deserialization(format!("Invalid entity ID: {}", e)))?;
            let to_id = EntityId::parse(&edge.to_entity_id)
                .map_err(|e| crate::Error::Deserialization(format!("Invalid entity ID: {}", e)))?;

            graph.link_entity_to_entity(&from_id, &to_id, &edge.relation_type);
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

    #[test]
    fn test_save_and_load_relationship_edges() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let storage = Arc::new(StorageEngine::open(&path).unwrap());
        let mut graph = GraphManager::new();

        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);

        // Create entities via memory links (so entity nodes exist)
        let alice_id = crate::types::EntityId::new();
        let bob_id = crate::types::EntityId::new();
        graph.link_memory_to_entity(&m1, &alice_id);
        graph.link_memory_to_entity(&m2, &bob_id);

        // Create entity-to-entity relationship
        graph.link_entity_to_entity(&alice_id, &bob_id, "spouse");

        // Verify edges before save: 2 memory→entity + 1 entity→entity = 3
        let (_, _, edge_count) = graph.entity_graph_stats();
        assert_eq!(edge_count, 3);

        // Alice should see Bob as related
        let related = graph.get_related_entities(&alice_id);
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].0, bob_id);
        assert_eq!(related[0].1, "spouse");

        // Save
        save_graph(&graph, &storage).unwrap();

        // Load into new graph
        let mut graph2 = GraphManager::new();
        load_graph(&mut graph2, &storage).unwrap();

        // Verify entity-to-entity edge survived round-trip
        let related2 = graph2.get_related_entities(&alice_id);
        assert_eq!(related2.len(), 1);
        assert_eq!(related2[0].0, bob_id);
        assert_eq!(related2[0].1, "spouse");

        // Bob should also see Alice (bidirectional traversal)
        let bob_related = graph2.get_related_entities(&bob_id);
        assert_eq!(bob_related.len(), 1);
        assert_eq!(bob_related[0].0, alice_id);
    }

    #[test]
    fn test_backward_compat_no_relationship_key() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let storage = Arc::new(StorageEngine::open(&path).unwrap());
        let mut graph = GraphManager::new();

        let m1 = make_memory_id(1);
        let alice_id = crate::types::EntityId::new();
        graph.link_memory_to_entity(&m1, &alice_id);

        // Save WITHOUT any relationship edges (simulates old DB)
        // Only entity graph is saved, no relationship_graph key
        let entity_edges = get_entity_edges(&graph);
        let entity_edge_data: Vec<EntityEdgeData> = entity_edges
            .into_iter()
            .map(|(memory_id, entity_id)| EntityEdgeData {
                memory_id: memory_id.to_string(),
                entity_id: entity_id.to_string(),
            })
            .collect();
        let entity_json = serde_json::to_vec(&entity_edge_data).unwrap();
        storage.store_entity_graph(&entity_json).unwrap();
        // Deliberately do NOT store relationship_graph

        // Load should not crash — absent key = empty
        let mut graph2 = GraphManager::new();
        load_graph(&mut graph2, &storage).unwrap();

        // Memory→entity edge should be present
        let memories = graph2.get_entity_memories(&alice_id);
        assert_eq!(memories.memories.len(), 1);

        // No relationship edges
        let related = graph2.get_related_entities(&alice_id);
        assert!(related.is_empty());
    }

    #[test]
    fn test_parse_relationship_target_of_pattern() {
        let names = vec!["mel".to_string(), "caroline".to_string()];
        // Standard "X of Y" format (Qwen3-4B)
        let result = parse_relationship_target("friend of Mel", &names);
        assert_eq!(result, Some(("friend", "Mel")));

        let result = parse_relationship_target("spouse of Caroline", &names);
        assert_eq!(result, Some(("spouse", "Caroline")));
    }

    #[test]
    fn test_parse_relationship_target_with_pattern() {
        let names = vec!["caroline".to_string()];
        // "X with Y" format (Qwen3-4B alternate)
        let result = parse_relationship_target("friendship with Caroline", &names);
        assert_eq!(result, Some(("friendship", "Caroline")));
    }

    #[test]
    fn test_parse_relationship_target_parenthetical() {
        let names = vec!["melanie".to_string(), "caroline".to_string(), "mel".to_string()];
        // "Entity (relation)" format (Phi-4-mini)
        let result = parse_relationship_target("Melanie (colleague or friend)", &names);
        assert_eq!(result, Some(("colleague or friend", "Melanie")));

        let result = parse_relationship_target("Caroline (friend or family)", &names);
        assert_eq!(result, Some(("friend or family", "Caroline")));

        let result = parse_relationship_target("Mel (friend or family member)", &names);
        assert_eq!(result, Some(("friend or family member", "Mel")));
    }

    #[test]
    fn test_parse_relationship_target_parenthetical_unknown_entity() {
        let names = vec!["caroline".to_string()];
        // Entity not in known_names — should return None
        let result = parse_relationship_target("UnknownPerson (friend)", &names);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_relationship_target_no_match() {
        let names = vec!["caroline".to_string()];
        // No recognized pattern
        assert_eq!(parse_relationship_target("just a string", &names), None);
        assert_eq!(parse_relationship_target("", &names), None);
        assert_eq!(parse_relationship_target("()", &names), None);
    }
}
