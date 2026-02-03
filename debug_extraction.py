#!/usr/bin/env python3
"""
Quick debug script to test LLM extraction output
Shows exactly what JSON the model returns
"""

import json
import subprocess
import sys

# Test with simple direct Rust code that calls the extractor
test_rust = """
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
"""

# Write test file
test_file = "/c/Users/georg/Documents/projects_new/claude_lz/mnemefusion/mnemefusion-core/examples/debug_extraction.rs"
print(f"[*] Writing test to {test_file}...", file=sys.stderr)
with open(test_file, "w") as f:
    f.write(test_rust)

# Build
print("[*] Building...", file=sys.stderr)
result = subprocess.run(
    ["cargo", "build", "--release", "--features", "entity-extraction", "--example", "debug_extraction", "-p", "mnemefusion-core"],
    cwd="/c/Users/georg/Documents/projects_new/claude_lz/mnemefusion",
    capture_output=True,
    timeout=600,
    text=True
)

if result.returncode != 0:
    print(f"[ERROR] Build failed:\n{result.stderr}", file=sys.stderr)
    sys.exit(1)

# Run
print("[*] Running extraction test...", file=sys.stderr)
result = subprocess.run(
    ["cargo", "run", "--release", "--features", "entity-extraction", "--example", "debug_extraction", "-p", "mnemefusion-core"],
    cwd="/c/Users/georg/Documents/projects_new/claude_lz/mnemefusion",
    capture_output=True,
    timeout=300,
    text=True
)

# Parse output
print("[*] Result:", file=sys.stderr)
print("STDERR:", file=sys.stderr)
for line in result.stderr.split("\n"):
    if "[" in line:  # Debug output
        print(f"  {line}", file=sys.stderr)

print("\nJSON OUTPUT:", file=sys.stderr)
print("=" * 80, file=sys.stderr)

# Extract JSON from stdout
try:
    # Try to parse as JSON
    output_lines = [l for l in result.stdout.split("\n") if l.strip()]
    if output_lines:
        json_str = "\n".join(output_lines)
        data = json.loads(json_str)

        print("PARSED SUCCESSFULLY", file=sys.stderr)
        print(f"  entities: {len(data.get('entities', []))} found", file=sys.stderr)
        print(f"  entity_facts: {len(data.get('entity_facts', []))} found", file=sys.stderr)
        print(f"  topics: {len(data.get('topics', []))} found", file=sys.stderr)

        # Pretty print
        print(json.dumps(data, indent=2))

        # Show the critical field
        print("\n" + "=" * 80, file=sys.stderr)
        print("[CRITICAL] Entity Facts (THIS IS WHAT WE'RE CHECKING):", file=sys.stderr)
        entity_facts = data.get('entity_facts', [])
        if entity_facts:
            print(f"  ✓ GOOD: Found {len(entity_facts)} facts", file=sys.stderr)
            for fact in entity_facts:
                print(f"    - {fact['entity']}: {fact['fact_type']} = {fact['value']}", file=sys.stderr)
        else:
            print(f"  ✗ BAD: entity_facts is EMPTY", file=sys.stderr)
            print(f"    This is the problem! The LLM is not returning facts.", file=sys.stderr)
    else:
        print("No output produced", file=sys.stderr)

except json.JSONDecodeError as e:
    print(f"[ERROR] JSON Parse Error: {e}", file=sys.stderr)
    print(f"Output was:\n{result.stdout}", file=sys.stderr)
    sys.exit(1)
