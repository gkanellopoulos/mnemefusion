//! Entity graph operations
//!
//! Manages relationships between memories and entities.
//! Entities are named concepts (people, organizations, projects) that
//! appear across multiple memories.

use crate::types::{EntityId, MemoryId};
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

    /// Create a new entity-to-entity relationship edge
    pub fn entity_relation(relation_type: &str) -> Self {
        Self {
            relationship: relation_type.to_string(),
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
        self.graph
            .add_edge(memory_idx, entity_idx, EntityEdge::mentions());
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
            for edge_ref in self
                .graph
                .edges_directed(entity_idx, petgraph::Direction::Incoming)
            {
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
            for edge_ref in self
                .graph
                .edges_directed(memory_idx, petgraph::Direction::Outgoing)
            {
                let target_idx = edge_ref.target();
                if let Some(EntityNode::Entity(entity_id)) = self.graph.node_weight(target_idx) {
                    entities.push(entity_id.clone());
                }
            }
        }

        entities
    }

    /// Link two entities with a relationship (entity-to-entity edge).
    ///
    /// Creates a directed edge from entity_a → entity_b with the given relation type.
    /// For bidirectional relationships (e.g., "spouse"), call this twice with swapped args.
    ///
    /// # Arguments
    ///
    /// * `from_entity_id` - The source entity
    /// * `to_entity_id` - The target entity
    /// * `relation_type` - Relationship type (e.g., "spouse", "colleague", "sibling")
    pub fn link_entity_to_entity(
        &mut self,
        from_entity_id: &EntityId,
        to_entity_id: &EntityId,
        relation_type: &str,
    ) {
        let from_idx = self.get_or_create_entity_node(from_entity_id);
        let to_idx = self.get_or_create_entity_node(to_entity_id);

        // Check if this exact edge already exists to avoid duplicates
        let exists = self
            .graph
            .edges_directed(from_idx, petgraph::Direction::Outgoing)
            .any(|e| e.target() == to_idx && e.weight().relationship == relation_type);

        if !exists {
            self.graph
                .add_edge(from_idx, to_idx, EntityEdge::entity_relation(relation_type));
        }
    }

    /// Get entities related to a given entity via entity-to-entity edges.
    ///
    /// Returns a list of (EntityId, relationship_type) tuples for all 1-hop
    /// entity-to-entity relationships (both outgoing and incoming).
    ///
    /// # Arguments
    ///
    /// * `entity_id` - The entity to find relationships for
    pub fn get_related_entities(&self, entity_id: &EntityId) -> Vec<(EntityId, String)> {
        let mut related = Vec::new();

        if let Some(&entity_idx) = self.entity_nodes.get(entity_id) {
            // Outgoing entity-to-entity edges
            for edge_ref in self
                .graph
                .edges_directed(entity_idx, petgraph::Direction::Outgoing)
            {
                let target_idx = edge_ref.target();
                if let Some(EntityNode::Entity(target_id)) = self.graph.node_weight(target_idx) {
                    // Skip "mentions" edges (those are memory→entity)
                    if edge_ref.weight().relationship != "mentions" {
                        related.push((target_id.clone(), edge_ref.weight().relationship.clone()));
                    }
                }
            }

            // Incoming entity-to-entity edges (bidirectional traversal)
            for edge_ref in self
                .graph
                .edges_directed(entity_idx, petgraph::Direction::Incoming)
            {
                let source_idx = edge_ref.source();
                if let Some(EntityNode::Entity(source_id)) = self.graph.node_weight(source_idx) {
                    if edge_ref.weight().relationship != "mentions" {
                        related.push((source_id.clone(), edge_ref.weight().relationship.clone()));
                    }
                }
            }
        }

        related
    }

    /// Remove all links for a memory (when memory is deleted)
    ///
    /// Note: Rebuilds node index maps since petgraph's remove_node
    /// can invalidate existing NodeIndex values.
    pub fn remove_memory(&mut self, memory_id: &MemoryId) {
        if let Some(&idx) = self.memory_nodes.get(memory_id) {
            self.graph.remove_node(idx);

            // Rebuild the node maps to ensure consistency
            self.rebuild_node_maps();
        }
    }

    /// Remove all links for an entity (when entity is deleted)
    ///
    /// Note: Rebuilds node index maps since petgraph's remove_node
    /// can invalidate existing NodeIndex values.
    pub fn remove_entity(&mut self, entity_id: &EntityId) {
        if let Some(&idx) = self.entity_nodes.get(entity_id) {
            self.graph.remove_node(idx);

            // Rebuild the node maps to ensure consistency
            self.rebuild_node_maps();
        }
    }

    /// Rebuild node maps after node removal
    ///
    /// This is necessary because petgraph's remove_node() can invalidate
    /// existing NodeIndex values when nodes are swapped.
    fn rebuild_node_maps(&mut self) {
        self.memory_nodes.clear();
        self.entity_nodes.clear();

        for node_idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(node_idx) {
                match node {
                    EntityNode::Memory(memory_id) => {
                        self.memory_nodes.insert(memory_id.clone(), node_idx);
                    }
                    EntityNode::Entity(entity_id) => {
                        self.entity_nodes.insert(entity_id.clone(), node_idx);
                    }
                }
            }
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
    fn test_entity_to_entity_link() {
        let mut graph = EntityGraph::new();

        let alice = Entity::new("Alice");
        let bob = Entity::new("Bob");

        graph.link_entity_to_entity(&alice.id, &bob.id, "spouse");

        // Should have 2 entity nodes and 1 edge
        assert_eq!(graph.entity_count(), 2);
        assert_eq!(graph.edge_count(), 1);

        // Alice should see Bob as related
        let alice_related = graph.get_related_entities(&alice.id);
        assert_eq!(alice_related.len(), 1);
        assert_eq!(alice_related[0].0, bob.id);
        assert_eq!(alice_related[0].1, "spouse");

        // Bob should also see Alice (incoming direction)
        let bob_related = graph.get_related_entities(&bob.id);
        assert_eq!(bob_related.len(), 1);
        assert_eq!(bob_related[0].0, alice.id);
        assert_eq!(bob_related[0].1, "spouse");
    }

    #[test]
    fn test_entity_to_entity_no_duplicate_edges() {
        let mut graph = EntityGraph::new();

        let alice = Entity::new("Alice");
        let bob = Entity::new("Bob");

        // Link twice with same relation — should not create duplicate
        graph.link_entity_to_entity(&alice.id, &bob.id, "colleague");
        graph.link_entity_to_entity(&alice.id, &bob.id, "colleague");

        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_entity_to_entity_multiple_relations() {
        let mut graph = EntityGraph::new();

        let alice = Entity::new("Alice");
        let bob = Entity::new("Bob");
        let carol = Entity::new("Carol");

        graph.link_entity_to_entity(&alice.id, &bob.id, "spouse");
        graph.link_entity_to_entity(&alice.id, &carol.id, "colleague");

        let alice_related = graph.get_related_entities(&alice.id);
        assert_eq!(alice_related.len(), 2);
    }

    #[test]
    fn test_entity_to_entity_excludes_mentions() {
        let mut graph = EntityGraph::new();

        let memory_id = MemoryId::new();
        let alice = Entity::new("Alice");
        let bob = Entity::new("Bob");

        // Memory mentions Alice (mentions edge)
        graph.link_memory_to_entity(&memory_id, &alice.id);
        // Alice related to Bob (entity-to-entity edge)
        graph.link_entity_to_entity(&alice.id, &bob.id, "friend");

        // get_related_entities should only return entity-to-entity links, not mentions
        let alice_related = graph.get_related_entities(&alice.id);
        assert_eq!(alice_related.len(), 1);
        assert_eq!(alice_related[0].0, bob.id);
        assert_eq!(alice_related[0].1, "friend");
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
