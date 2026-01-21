//! Causal graph implementation using petgraph.
//!
//! The causal graph tracks cause-effect relationships between memories:
//! - Each node is a MemoryId
//! - Each edge is a CausalEdge (confidence + evidence)
//! - Supports multi-hop traversal with configurable depth

use crate::{types::{EntityId, MemoryId}, Error, Result};
use super::entity::{EntityGraph, EntityQueryResult};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::HashMap;

/// A causal edge linking two memories.
#[derive(Debug, Clone)]
pub struct CausalEdge {
    /// Confidence score for this causal link (0.0 to 1.0)
    pub confidence: f32,
    /// Evidence text explaining why this is a causal relationship
    pub evidence: String,
}

impl CausalEdge {
    /// Create a new causal edge.
    pub fn new(confidence: f32, evidence: String) -> Result<Self> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(Error::InvalidParameter(
                "Confidence must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self {
            confidence,
            evidence,
        })
    }
}

/// A path in the causal graph from one memory to another.
#[derive(Debug, Clone)]
pub struct CausalPath {
    /// Sequence of memory IDs from start to end
    pub memories: Vec<MemoryId>,
    /// Cumulative confidence (product of edge confidences along path)
    pub confidence: f32,
}

/// Result of causal traversal (get_causes or get_effects).
#[derive(Debug, Clone)]
pub struct CausalTraversalResult {
    /// All paths found within max_hops
    pub paths: Vec<CausalPath>,
}

/// Manages the causal and entity graph structures.
pub struct GraphManager {
    /// The directed graph (cause → effect)
    graph: DiGraph<MemoryId, CausalEdge>,
    /// Maps MemoryId to NodeIndex for efficient lookup
    node_map: HashMap<MemoryId, NodeIndex>,
    /// Entity graph (memory ↔ entity relationships)
    entity_graph: EntityGraph,
}

impl GraphManager {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            entity_graph: EntityGraph::new(),
        }
    }

    /// Get or create a node for a memory ID.
    fn get_or_create_node(&mut self, memory_id: &MemoryId) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(memory_id) {
            return idx;
        }

        let idx = self.graph.add_node(memory_id.clone());
        self.node_map.insert(memory_id.clone(), idx);
        idx
    }

    /// Add a causal link: cause → effect.
    ///
    /// # Arguments
    /// * `cause` - The MemoryId of the cause
    /// * `effect` - The MemoryId of the effect
    /// * `confidence` - Confidence score (0.0 to 1.0)
    /// * `evidence` - Evidence text explaining the causal relationship
    ///
    /// # Errors
    /// Returns error if confidence is not in range [0.0, 1.0]
    pub fn add_causal_link(
        &mut self,
        cause: &MemoryId,
        effect: &MemoryId,
        confidence: f32,
        evidence: String,
    ) -> Result<()> {
        let edge = CausalEdge::new(confidence, evidence)?;

        let cause_node = self.get_or_create_node(cause);
        let effect_node = self.get_or_create_node(effect);

        // Add edge from cause to effect
        self.graph.add_edge(cause_node, effect_node, edge);

        Ok(())
    }

    /// Get causes of a memory (backward traversal).
    ///
    /// Uses BFS to traverse backward (incoming edges) up to max_hops.
    ///
    /// # Arguments
    /// * `memory_id` - The memory to find causes for
    /// * `max_hops` - Maximum traversal depth
    ///
    /// # Returns
    /// CausalTraversalResult with all paths found
    pub fn get_causes(&self, memory_id: &MemoryId, max_hops: usize) -> Result<CausalTraversalResult> {
        let start_node = self
            .node_map
            .get(memory_id)
            .ok_or_else(|| Error::MemoryNotFound(memory_id.to_string()))?;

        let mut paths = Vec::new();

        // BFS with depth tracking
        self.traverse_paths(*start_node, max_hops, Direction::Incoming, &mut paths);

        Ok(CausalTraversalResult { paths })
    }

    /// Get effects of a memory (forward traversal).
    ///
    /// Uses BFS to traverse forward (outgoing edges) up to max_hops.
    ///
    /// # Arguments
    /// * `memory_id` - The memory to find effects for
    /// * `max_hops` - Maximum traversal depth
    ///
    /// # Returns
    /// CausalTraversalResult with all paths found
    pub fn get_effects(&self, memory_id: &MemoryId, max_hops: usize) -> Result<CausalTraversalResult> {
        let start_node = self
            .node_map
            .get(memory_id)
            .ok_or_else(|| Error::MemoryNotFound(memory_id.to_string()))?;

        let mut paths = Vec::new();

        // BFS with depth tracking
        self.traverse_paths(*start_node, max_hops, Direction::Outgoing, &mut paths);

        Ok(CausalTraversalResult { paths })
    }

    /// Internal BFS traversal that builds paths with cumulative confidence.
    fn traverse_paths(
        &self,
        start: NodeIndex,
        max_hops: usize,
        direction: Direction,
        paths: &mut Vec<CausalPath>,
    ) {
        if max_hops == 0 {
            return;
        }

        // Queue: (current_node, path_so_far, cumulative_confidence, depth)
        let mut queue = vec![(
            start,
            vec![self.graph[start].clone()],
            1.0f32,
            0usize,
        )];
        let mut visited = HashMap::new();
        visited.insert(start, 0);

        while let Some((current, path, confidence, depth)) = queue.pop() {
            if depth >= max_hops {
                continue;
            }

            // Get neighbors in the specified direction
            let neighbors: Vec<_> = self
                .graph
                .neighbors_directed(current, direction)
                .collect();

            for neighbor in neighbors {
                // Check if we've visited this node at a shallower depth
                if let Some(&prev_depth) = visited.get(&neighbor) {
                    if prev_depth <= depth + 1 {
                        continue; // Skip if we've seen this node at same or shallower depth
                    }
                }
                visited.insert(neighbor, depth + 1);

                // Find the edge
                let edge = if direction == Direction::Outgoing {
                    self.graph.find_edge(current, neighbor)
                } else {
                    self.graph.find_edge(neighbor, current)
                };

                if let Some(edge_idx) = edge {
                    let edge_data = &self.graph[edge_idx];
                    let new_confidence = confidence * edge_data.confidence;

                    let mut new_path = path.clone();
                    new_path.push(self.graph[neighbor].clone());

                    // Add this as a valid path
                    paths.push(CausalPath {
                        memories: new_path.clone(),
                        confidence: new_confidence,
                    });

                    // Continue traversal
                    queue.push((neighbor, new_path, new_confidence, depth + 1));
                }
            }
        }
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if a memory exists in the graph.
    pub fn contains(&self, memory_id: &MemoryId) -> bool {
        self.node_map.contains_key(memory_id)
    }

    /// Get all edges (for persistence).
    pub(crate) fn edges(&self) -> Vec<(MemoryId, MemoryId, CausalEdge)> {
        let mut result = Vec::new();
        for edge in self.graph.edge_references() {
            let source = self.graph[edge.source()].clone();
            let target = self.graph[edge.target()].clone();
            let data = edge.weight().clone();
            result.push((source, target, data));
        }
        result
    }

    /// Load edges into the graph (for persistence).
    ///
    /// Clears the current graph and rebuilds it from the provided edges.
    pub(crate) fn load_edges(&mut self, edges: Vec<(MemoryId, MemoryId, CausalEdge)>) -> Result<()> {
        self.graph.clear();
        self.node_map.clear();

        for (source, target, edge_data) in edges {
            self.add_causal_link(&source, &target, edge_data.confidence, edge_data.evidence)?;
        }

        Ok(())
    }

    /// Clear the graph (for testing).
    #[cfg(test)]
    pub fn clear(&mut self) {
        self.graph.clear();
        self.node_map.clear();
        self.entity_graph.clear();
    }

    // ========== Entity Graph Operations ==========

    /// Link a memory to an entity
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory that mentions the entity
    /// * `entity_id` - The entity being mentioned
    pub fn link_memory_to_entity(&mut self, memory_id: &MemoryId, entity_id: &EntityId) {
        self.entity_graph.link_memory_to_entity(memory_id, entity_id);
    }

    /// Get all memories that mention a specific entity
    ///
    /// # Arguments
    ///
    /// * `entity_id` - The entity to query
    ///
    /// # Returns
    ///
    /// EntityQueryResult containing list of memory IDs
    pub fn get_entity_memories(&self, entity_id: &EntityId) -> EntityQueryResult {
        self.entity_graph.get_entity_memories(entity_id)
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
        self.entity_graph.get_memory_entities(memory_id)
    }

    /// Remove a memory from the entity graph (called when memory is deleted)
    pub fn remove_memory_from_entity_graph(&mut self, memory_id: &MemoryId) {
        self.entity_graph.remove_memory(memory_id);
    }

    /// Remove an entity from the entity graph (called when entity is deleted)
    pub fn remove_entity_from_graph(&mut self, entity_id: &EntityId) {
        self.entity_graph.remove_entity(entity_id);
    }

    /// Remove a memory from the causal graph (called when memory is deleted)
    ///
    /// This removes all causal links (both incoming and outgoing) associated
    /// with the specified memory.
    pub fn remove_memory_from_causal_graph(&mut self, memory_id: &MemoryId) {
        // Find the node for this memory
        if let Some(&node_idx) = self.node_map.get(memory_id) {
            // Remove the node (this also removes all edges)
            self.graph.remove_node(node_idx);

            // Remove from the node map
            self.node_map.remove(memory_id);

            // Note: NodeIndex values may have changed after remove_node
            // Rebuild the node_map to ensure consistency
            let mut new_map = HashMap::new();
            for node_idx in self.graph.node_indices() {
                if let Some(mem_id) = self.graph.node_weight(node_idx) {
                    new_map.insert(mem_id.clone(), node_idx);
                }
            }
            self.node_map = new_map;
        }
    }

    /// Get entity graph statistics
    pub fn entity_graph_stats(&self) -> (usize, usize, usize) {
        (
            self.entity_graph.memory_count(),
            self.entity_graph.entity_count(),
            self.entity_graph.edge_count(),
        )
    }

    /// Get a reference to the entity graph (for persistence)
    pub(super) fn entity_graph(&self) -> &EntityGraph {
        &self.entity_graph
    }

    /// Get a mutable reference to the entity graph (for persistence)
    pub(super) fn entity_graph_mut(&mut self) -> &mut EntityGraph {
        &mut self.entity_graph
    }
}

impl Default for GraphManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory_id(n: u128) -> MemoryId {
        MemoryId::from_u128(n)
    }

    #[test]
    fn test_create_empty_graph() {
        let graph = GraphManager::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_single_causal_link() {
        let mut graph = GraphManager::new();
        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);

        graph
            .add_causal_link(&m1, &m2, 0.8, "m1 caused m2".to_string())
            .unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert!(graph.contains(&m1));
        assert!(graph.contains(&m2));
    }

    #[test]
    fn test_invalid_confidence() {
        let mut graph = GraphManager::new();
        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);

        // Confidence > 1.0
        let result = graph.add_causal_link(&m1, &m2, 1.5, "invalid".to_string());
        assert!(result.is_err());

        // Confidence < 0.0
        let result = graph.add_causal_link(&m1, &m2, -0.1, "invalid".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_get_causes_single_hop() {
        let mut graph = GraphManager::new();
        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);

        // m1 → m2
        graph
            .add_causal_link(&m1, &m2, 0.9, "m1 caused m2".to_string())
            .unwrap();

        // Get causes of m2 (should find m1)
        let result = graph.get_causes(&m2, 1).unwrap();
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].memories.len(), 2);
        assert_eq!(result.paths[0].memories[0], m2);
        assert_eq!(result.paths[0].memories[1], m1);
        assert!((result.paths[0].confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_get_effects_single_hop() {
        let mut graph = GraphManager::new();
        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);

        // m1 → m2
        graph
            .add_causal_link(&m1, &m2, 0.8, "m1 caused m2".to_string())
            .unwrap();

        // Get effects of m1 (should find m2)
        let result = graph.get_effects(&m1, 1).unwrap();
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].memories.len(), 2);
        assert_eq!(result.paths[0].memories[0], m1);
        assert_eq!(result.paths[0].memories[1], m2);
        assert!((result.paths[0].confidence - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_multi_hop_chain() {
        let mut graph = GraphManager::new();
        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);
        let m3 = make_memory_id(3);
        let m4 = make_memory_id(4);

        // Chain: m1 → m2 → m3 → m4
        graph
            .add_causal_link(&m1, &m2, 0.9, "m1→m2".to_string())
            .unwrap();
        graph
            .add_causal_link(&m2, &m3, 0.8, "m2→m3".to_string())
            .unwrap();
        graph
            .add_causal_link(&m3, &m4, 0.7, "m3→m4".to_string())
            .unwrap();

        // Get effects of m1 with max_hops=3
        let result = graph.get_effects(&m1, 3).unwrap();

        // Should find 3 paths: m1→m2, m1→m2→m3, m1→m2→m3→m4
        assert_eq!(result.paths.len(), 3);

        // Verify cumulative confidence for longest path
        let longest_path = result
            .paths
            .iter()
            .max_by_key(|p| p.memories.len())
            .unwrap();
        assert_eq!(longest_path.memories.len(), 4);
        let expected_conf = 0.9 * 0.8 * 0.7;
        assert!((longest_path.confidence - expected_conf).abs() < 0.001);
    }

    #[test]
    fn test_branching_graph() {
        let mut graph = GraphManager::new();
        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);
        let m3 = make_memory_id(3);

        // m1 → m2, m1 → m3 (m1 has two effects)
        graph
            .add_causal_link(&m1, &m2, 0.9, "m1→m2".to_string())
            .unwrap();
        graph
            .add_causal_link(&m1, &m3, 0.8, "m1→m3".to_string())
            .unwrap();

        // Get effects of m1
        let result = graph.get_effects(&m1, 1).unwrap();

        // Should find 2 paths
        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn test_max_hops_limit() {
        let mut graph = GraphManager::new();
        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);
        let m3 = make_memory_id(3);

        // m1 → m2 → m3
        graph
            .add_causal_link(&m1, &m2, 0.9, "m1→m2".to_string())
            .unwrap();
        graph
            .add_causal_link(&m2, &m3, 0.8, "m2→m3".to_string())
            .unwrap();

        // Get effects with max_hops=1 (should only find m2)
        let result = graph.get_effects(&m1, 1).unwrap();
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].memories.len(), 2);

        // Get effects with max_hops=2 (should find m2 and m3)
        let result = graph.get_effects(&m1, 2).unwrap();
        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn test_disconnected_node() {
        let graph = GraphManager::new();
        let m1 = make_memory_id(1);

        // Try to get causes for non-existent node
        let result = graph.get_causes(&m1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_edges_for_persistence() {
        let mut graph = GraphManager::new();
        let m1 = make_memory_id(1);
        let m2 = make_memory_id(2);
        let m3 = make_memory_id(3);

        graph
            .add_causal_link(&m1, &m2, 0.9, "edge1".to_string())
            .unwrap();
        graph
            .add_causal_link(&m2, &m3, 0.8, "edge2".to_string())
            .unwrap();

        let edges = graph.edges();
        assert_eq!(edges.len(), 2);

        // Verify we can reconstruct the graph
        let mut new_graph = GraphManager::new();
        for (source, target, edge_data) in edges {
            new_graph
                .add_causal_link(&source, &target, edge_data.confidence, edge_data.evidence)
                .unwrap();
        }

        assert_eq!(new_graph.node_count(), 3);
        assert_eq!(new_graph.edge_count(), 2);
    }
}
