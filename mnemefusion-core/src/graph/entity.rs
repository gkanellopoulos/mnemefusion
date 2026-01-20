//! Entity graph operations
//!
//! Manages relationships between memories and entities.
//! Entities are named concepts (people, organizations, projects) that
//! appear across multiple memories.

use crate::{
    types::{Entity, EntityId, MemoryId},
    Result,
};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Node type in the entity graph
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityNode {
    /// A memory node
    Memory(MemoryId),
    /// An entity node
    Entity(EntityId),
}

/// Edge representing a relationship between memory and entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEdge {
    /// Relationship type (currently always "mentions")
    pub relationship: String,
}

impl EntityEdge {
    /// Create a new entity edge with "mentions" relationship
    pub fn mentions() -> Self {
        Self {
            relationship: "mentions".to_string(),
        }
    }
}

/// Result from entity graph queries
#[derive(Debug, Clone)]
pub struct EntityQueryResult {
    /// List of memory IDs that relate to the queried entity
    pub memories: Vec<MemoryId>,
}

/// Manages the entity graph
///
/// The entity graph is a bipartite graph connecting memories to entities.
/// Edges represent "mentions" relationships (memory mentions entity).
pub struct EntityGraph {
    /// The underlying graph structure
    pub(super) graph: DiGraph<EntityNode, EntityEdge>,

    /// Map from EntityId to NodeIndex for O(1) lookup
    entity_nodes: HashMap<EntityId, NodeIndex>,

    /// Map from MemoryId to NodeIndex for O(1) lookup
    memory_nodes: HashMap<MemoryId, NodeIndex>,
}

impl EntityGraph {
    /// Create a new empty entity graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            entity_nodes: HashMap::new(),
            memory_nodes: HashMap::new(),
        }
    }

    /// Add or get a memory node
    fn get_or_create_memory_node(&mut self, memory_id: &MemoryId) -> NodeIndex {
        if let Some(&idx) = self.memory_nodes.get(memory_id) {
            idx
        } else {
            let idx = self.graph.add_node(EntityNode::Memory(memory_id.clone()));
            self.memory_nodes.insert(memory_id.clone(), idx);
            idx
        }
    }

    /// Add or get an entity node
    fn get_or_create_entity_node(&mut self, entity_id: &EntityId) -> NodeIndex {
        if let Some(&idx) = self.entity_nodes.get(entity_id) {
            idx
        } else {
            let idx = self.graph.add_node(EntityNode::Entity(entity_id.clone()));
            self.entity_nodes.insert(entity_id.clone(), idx);
            idx
        }
    }

    /// Link a memory to an entity
    ///
    /// Creates a directed edge from memory → entity with "mentions" relationship.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory that mentions the entity
    /// * `entity_id` - The entity being mentioned
    pub fn link_memory_to_entity(&mut self, memory_id: &MemoryId, entity_id: &EntityId) {
        let memory_idx = self.get_or_create_memory_node(memory_id);
        let entity_idx = self.get_or_create_entity_node(entity_id);

        // Add edge from memory to entity
        self.graph.add_edge(memory_idx, entity_idx, EntityEdge::mentions());
    }

    /// Get all memories that mention a specific entity
    ///
    /// # Arguments
    ///
    /// * `entity_id` - The entity to query
    ///
    /// # Returns
    ///
    /// List of memory IDs that mention this entity
    pub fn get_entity_memories(&self, entity_id: &EntityId) -> EntityQueryResult {
        let mut memories = Vec::new();

        if let Some(&entity_idx) = self.entity_nodes.get(entity_id) {
            // Find all edges pointing to this entity
            for edge_ref in self.graph.edges_directed(entity_idx, petgraph::Direction::Incoming) {
                let source_idx = edge_ref.source();
                if let Some(EntityNode::Memory(memory_id)) = self.graph.node_weight(source_idx) {
                    memories.push(memory_id.clone());
                }
            }
        }

        EntityQueryResult { memories }
    }

    /// Get all entities mentioned in a specific memory
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory to query
    ///
    /// # Returns
    ///
    /// List of entity IDs mentioned in this memory
    pub fn get_memory_entities(&self, memory_id: &MemoryId) -> Vec<EntityId> {
        let mut entities = Vec::new();

        if let Some(&memory_idx) = self.memory_nodes.get(memory_id) {
            // Find all edges from this memory
            for edge_ref in self.graph.edges_directed(memory_idx, petgraph::Direction::Outgoing) {
                let target_idx = edge_ref.target();
                if let Some(EntityNode::Entity(entity_id)) = self.graph.node_weight(target_idx) {
                    entities.push(entity_id.clone());
                }
            }
        }

        entities
    }

    /// Remove all links for a memory (when memory is deleted)
    pub fn remove_memory(&mut self, memory_id: &MemoryId) {
        if let Some(&idx) = self.memory_nodes.get(memory_id) {
            self.graph.remove_node(idx);
            self.memory_nodes.remove(memory_id);
        }
    }

    /// Remove all links for an entity (when entity is deleted)
    pub fn remove_entity(&mut self, entity_id: &EntityId) {
        if let Some(&idx) = self.entity_nodes.get(entity_id) {
            self.graph.remove_node(idx);
            self.entity_nodes.remove(entity_id);
        }
    }

    /// Get the number of memories in the graph
    pub fn memory_count(&self) -> usize {
        self.memory_nodes.len()
    }

    /// Get the number of entities in the graph
    pub fn entity_count(&self) -> usize {
        self.entity_nodes.len()
    }

    /// Get the total number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Clear all nodes and edges
    pub fn clear(&mut self) {
        self.graph.clear();
        self.memory_nodes.clear();
        self.entity_nodes.clear();
    }
}

impl Default for EntityGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Entity;

    #[test]
    fn test_entity_graph_new() {
        let graph = EntityGraph::new();
        assert_eq!(graph.memory_count(), 0);
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_entity_graph_link() {
        let mut graph = EntityGraph::new();

        let memory_id = MemoryId::new();
        let entity = Entity::new("Project Alpha");
        let entity_id = entity.id.clone();

        graph.link_memory_to_entity(&memory_id, &entity_id);

        assert_eq!(graph.memory_count(), 1);
        assert_eq!(graph.entity_count(), 1);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_entity_graph_get_entity_memories() {
        let mut graph = EntityGraph::new();

        let mem1 = MemoryId::new();
        let mem2 = MemoryId::new();
        let mem3 = MemoryId::new();

        let entity = Entity::new("Alice");
        let entity_id = entity.id.clone();

        // Link memories to entity
        graph.link_memory_to_entity(&mem1, &entity_id);
        graph.link_memory_to_entity(&mem2, &entity_id);
        graph.link_memory_to_entity(&mem3, &entity_id);

        // Query entity memories
        let result = graph.get_entity_memories(&entity_id);
        assert_eq!(result.memories.len(), 3);
        assert!(result.memories.contains(&mem1));
        assert!(result.memories.contains(&mem2));
        assert!(result.memories.contains(&mem3));
    }

    #[test]
    fn test_entity_graph_get_memory_entities() {
        let mut graph = EntityGraph::new();

        let memory_id = MemoryId::new();

        let entity1 = Entity::new("Alice");
        let entity2 = Entity::new("Project Alpha");
        let entity3 = Entity::new("Q1");

        // Link memory to multiple entities
        graph.link_memory_to_entity(&memory_id, &entity1.id);
        graph.link_memory_to_entity(&memory_id, &entity2.id);
        graph.link_memory_to_entity(&memory_id, &entity3.id);

        // Query memory entities
        let entities = graph.get_memory_entities(&memory_id);
        assert_eq!(entities.len(), 3);
        assert!(entities.contains(&entity1.id));
        assert!(entities.contains(&entity2.id));
        assert!(entities.contains(&entity3.id));
    }

    #[test]
    fn test_entity_graph_remove_memory() {
        let mut graph = EntityGraph::new();

        let memory_id = MemoryId::new();
        let entity = Entity::new("Test");

        graph.link_memory_to_entity(&memory_id, &entity.id);
        assert_eq!(graph.edge_count(), 1);

        graph.remove_memory(&memory_id);
        assert_eq!(graph.memory_count(), 0);
        assert_eq!(graph.edge_count(), 0);

        // Entity node should still exist
        assert_eq!(graph.entity_count(), 1);
    }

    #[test]
    fn test_entity_graph_remove_entity() {
        let mut graph = EntityGraph::new();

        let memory_id = MemoryId::new();
        let entity = Entity::new("Test");

        graph.link_memory_to_entity(&memory_id, &entity.id);
        assert_eq!(graph.edge_count(), 1);

        graph.remove_entity(&entity.id);
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.edge_count(), 0);

        // Memory node should still exist
        assert_eq!(graph.memory_count(), 1);
    }

    #[test]
    fn test_entity_graph_multiple_links() {
        let mut graph = EntityGraph::new();

        let mem1 = MemoryId::new();
        let mem2 = MemoryId::new();
        let entity1 = Entity::new("Alice");
        let entity2 = Entity::new("Bob");

        // Create a small network
        graph.link_memory_to_entity(&mem1, &entity1.id);
        graph.link_memory_to_entity(&mem1, &entity2.id);
        graph.link_memory_to_entity(&mem2, &entity1.id);

        assert_eq!(graph.memory_count(), 2);
        assert_eq!(graph.entity_count(), 2);
        assert_eq!(graph.edge_count(), 3);

        // Query Alice's memories
        let alice_memories = graph.get_entity_memories(&entity1.id);
        assert_eq!(alice_memories.memories.len(), 2);

        // Query Bob's memories
        let bob_memories = graph.get_entity_memories(&entity2.id);
        assert_eq!(bob_memories.memories.len(), 1);
    }
}
