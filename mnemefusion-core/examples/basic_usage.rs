//! Basic usage example for MnemeFusion
//!
//! This example demonstrates the core functionality:
//! - Creating/opening a database
//! - Adding memories
//! - Retrieving memories
//! - Using metadata
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

        let id = engine.add(content.to_string(), embedding, Some(metadata), None)?;
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

    // Delete a memory
    println!("\nDeleting the third memory...");
    let deleted = engine.delete(&memory_ids[2])?;
    if deleted {
        println!("  Successfully deleted");
        println!("  Remaining memories: {}", engine.count()?);
    }

    // Close the database
    println!("\nClosing database...");
    engine.close()?;

    println!("\n✓ Example completed successfully!");
    println!("The database file 'example_brain.mfdb' has been created.");
    println!("Run this example again to see persistence in action.");

    Ok(())
}
