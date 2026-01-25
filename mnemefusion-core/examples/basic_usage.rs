//! Basic usage example for MnemeFusion
//!
//! This example demonstrates the core functionality:
//! - Creating/opening a database
//! - Adding memories with embeddings
//! - Retrieving memories by ID
//! - Searching by semantic similarity
//! - Querying memories by time (recent, range)
//! - Adding causal relationships between memories
//! - Querying causal chains (causes and effects)
//! - Using metadata
//! - Deleting memories
//! - Persistence across restarts
//!
//! Run with: cargo run --example basic_usage

use mnemefusion_core::{Config, MemoryEngine};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MnemeFusion Basic Usage Example\n");

    // Open or create a database
    println!("Opening database at ./example_brain.mfdb");
    let engine = MemoryEngine::open("./example_brain.mfdb", Config::default())?;

    // Show initial count
    let initial_count = engine.count()?;
    println!("Initial memory count: {}\n", initial_count);

    // Add some memories with simple embeddings
    println!("Adding memories...");

    let memories = vec![
        "Project deadline moved to March 15th",
        "Team meeting scheduled for tomorrow at 2pm",
        "Budget approved for Q2 expansion",
        "New hire starts next Monday",
        "Code review completed successfully",
    ];

    let mut memory_ids = Vec::new();

    for (idx, content) in memories.iter().enumerate() {
        // In a real application, you'd use a proper embedding model
        // For this example, we'll use simple dummy embeddings
        let embedding = vec![(idx as f32) * 0.1; 384];

        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("example".to_string(), "basic_usage".to_string());
        metadata.insert("index".to_string(), idx.to_string());

        let id = engine.add(
            content.to_string(),
            embedding,
            Some(metadata),
            None,
            None,
            None,
        )?;
        println!("  Added: {} (ID: {}...)", content, &id.to_string()[..8]);
        memory_ids.push(id);
    }

    // Show updated count
    let count = engine.count()?;
    println!("\nTotal memories: {}", count);

    // Retrieve a specific memory
    println!("\nRetrieving first memory...");
    if let Some(memory) = engine.get(&memory_ids[0])? {
        println!("  Content: {}", memory.content);
        println!("  Embedding dimension: {}", memory.embedding.len());
        println!("  Metadata: {:?}", memory.metadata);
        println!(
            "  Created: {} seconds since epoch",
            memory.created_at.as_unix_secs()
        );
    }

    // List all memory IDs
    println!("\nAll memory IDs:");
    let all_ids = engine.list_ids()?;
    for id in &all_ids {
        println!("  {}", id);
    }

    // Search for similar memories
    println!("\nSearching for memories similar to 'meeting' topic...");
    // In a real application, you'd embed the query with the same model
    // For this example, we use a query vector similar to our second memory (team meeting)
    let query_embedding = vec![0.1; 384]; // Similar to idx=1 memory
    let results = engine.search(&query_embedding, 3, None, None)?;

    println!("  Found {} results:", results.len());
    for (idx, (memory, similarity)) in results.iter().enumerate() {
        println!(
            "    {}. [Similarity: {:.3}] {}",
            idx + 1,
            similarity,
            memory.content
        );
    }

    // Temporal queries
    println!("\nQuerying most recent memories...");
    let recent = engine.get_recent(3, None)?;
    println!("  3 most recent memories:");
    for (idx, (memory, timestamp)) in recent.iter().enumerate() {
        println!(
            "    {}. [{}] {}",
            idx + 1,
            timestamp.as_unix_secs() as u64,
            memory.content
        );
    }

    // Time range query
    use mnemefusion_core::Timestamp;
    let now = Timestamp::now();
    let one_hour_ago = now.subtract_days(0); // For demo, just use now as the range
    println!("\nQuerying memories from the last session...");
    let range_results = engine.get_range(one_hour_ago.subtract_days(1), now, 10, None)?;
    println!("  Found {} memories in time range", range_results.len());

    // Causal relationships
    println!("\nAdding causal relationships...");
    // memory_ids[0] = "Project deadline moved"
    // memory_ids[1] = "Team meeting scheduled"
    // memory_ids[2] = "Budget approved"

    // Create causal chain: deadline → meeting → budget
    engine.add_causal_link(
        &memory_ids[0],
        &memory_ids[1],
        0.85,
        "Deadline change triggered team meeting".to_string(),
    )?;

    engine.add_causal_link(
        &memory_ids[1],
        &memory_ids[2],
        0.75,
        "Meeting led to budget approval".to_string(),
    )?;

    println!("  Added 2 causal links");

    // Query causal effects
    println!("\nQuerying causal effects of deadline change...");
    let effects = engine.get_effects(&memory_ids[0], 2)?;
    println!("  Found {} causal paths:", effects.paths.len());
    for (idx, path) in effects.paths.iter().enumerate() {
        println!(
            "    {}. Path length: {}, confidence: {:.2}",
            idx + 1,
            path.memories.len(),
            path.confidence
        );
    }

    // Query causal causes
    println!("\nQuerying causes of budget approval...");
    let causes = engine.get_causes(&memory_ids[2], 2)?;
    println!("  Found {} causal paths backward:", causes.paths.len());
    for (idx, path) in causes.paths.iter().enumerate() {
        println!(
            "    {}. Path length: {}, confidence: {:.2}",
            idx + 1,
            path.memories.len(),
            path.confidence
        );
    }

    // Entity extraction and queries
    println!("\nEntity extraction (automatic from memory content)...");
    println!("  Entities were automatically extracted when memories were added");

    // List all entities
    let all_entities = engine.list_entities()?;
    println!("  Found {} unique entities:", all_entities.len());
    for entity in &all_entities {
        println!("    - {} ({} mentions)", entity.name, entity.mention_count);
    }

    // Query memories by entity
    if !all_entities.is_empty() {
        let example_entity = &all_entities[0];
        println!(
            "\nQuerying memories that mention '{}'...",
            example_entity.name
        );
        let entity_memories = engine.get_entity_memories(&example_entity.name)?;
        println!("  Found {} memories:", entity_memories.len());
        for (idx, memory) in entity_memories.iter().enumerate() {
            println!("    {}. {}", idx + 1, memory.content);
        }
    }

    // Query entities in a specific memory
    println!("\nQuerying entities in first memory...");
    let entities_in_memory = engine.get_memory_entities(&memory_ids[0])?;
    println!("  Found {} entities:", entities_in_memory.len());
    for entity in entities_in_memory {
        println!("    - {}", entity.name);
    }

    // Delete a memory
    println!("\nDeleting the third memory...");
    let deleted = engine.delete(&memory_ids[2], None)?;
    if deleted {
        println!("  Successfully deleted");
        println!("  Remaining memories: {}", engine.count()?);
    }

    // Close the database (saves all indexes)
    println!("\nClosing database...");
    engine.close()?;

    println!("\n✓ Example completed successfully!");
    println!("The database file 'example_brain.mfdb' has been created.");
    println!("Run this example again to see persistence in action.");

    Ok(())
}
