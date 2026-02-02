// Quick test for LLM extraction
// Run with: cargo run --release --features entity-extraction -p mnemefusion-core --example test_llm_extraction

use mnemefusion_core::extraction::{LlmEntityExtractor, ModelTier};

fn main() {
    println!("Loading Qwen3-4B model...");

    let extractor = match LlmEntityExtractor::load(ModelTier::Balanced) {
        Ok(e) => {
            println!("Model loaded successfully!");
            e
        }
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    println!("\nTesting extraction...");
    let test_cases = [
        "Caroline is researching adoption agencies to find her biological parents. She works as a counselor in Boston.",
        "Alice works as a software engineer at Google and lives in Seattle.",
        "Bob wants to learn machine learning and is taking online courses.",
    ];

    for (i, test_text) in test_cases.iter().enumerate() {
        println!("\n--- Test Case {} ---", i + 1);
        println!("Input: {}", test_text);

        match extractor.extract(test_text) {
            Ok(result) => {
                println!("Extraction Result:");
                println!("  Entities: {:?}", result.entities.iter().map(|e| &e.name).collect::<Vec<_>>());
                println!("  Entity Facts:");
                for fact in &result.entity_facts {
                    println!("    - {}: {} = {} (confidence: {:.2})",
                        fact.entity, fact.fact_type, fact.value, fact.confidence);
                }
                println!("  Topics: {:?}", result.topics);
                println!("  Importance: {:.2}", result.importance);
            }
            Err(e) => {
                eprintln!("Extraction failed: {}", e);
            }
        }
    }

    println!("\n=== All tests complete ===");
}
