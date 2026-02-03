// Debug script to test LLM extraction output
// Shows exactly what JSON the model returns
// Run with: cargo run --release --features entity-extraction -p mnemefusion-core --example debug_extraction

use mnemefusion_core::extraction::{LlmEntityExtractor, ModelTier};

fn main() {
    eprintln!("[DEBUG] Loading model...");
    let extractor = match LlmEntityExtractor::load(ModelTier::Balanced) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[ERROR] Failed to load: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!("[DEBUG] Model loaded, testing extraction...");

    let test_text = "Caroline is researching adoption agencies to find her biological parents. She works as a counselor in Boston.";

    match extractor.extract(test_text) {
        Ok(result) => {
            eprintln!("[DEBUG] Extraction succeeded!");
            eprintln!("[DEBUG] entity_facts count: {}", result.entity_facts.len());

            // Print raw JSON
            if let Ok(json_str) = serde_json::to_string_pretty(&result) {
                println!("{}", json_str);
            } else {
                eprintln!("[ERROR] Failed to serialize to JSON");
            }
        }
        Err(e) => {
            eprintln!("[ERROR] Extraction failed: {}", e);
            std::process::exit(1);
        }
    }
}
