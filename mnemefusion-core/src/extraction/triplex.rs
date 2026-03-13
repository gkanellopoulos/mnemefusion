//! Triplex KG triple extractor
//!
//! SciPhi Triplex is a Phi-3 3.8B fine-tune purpose-built for knowledge graph
//! construction. It extracts (subject, predicate, object) triples from text
//! with predefined entity types and relationship predicates.
//!
//! This module provides:
//! - `TriplexExtractor`: loads the Triplex GGUF model and runs inference
//! - `parse_triplex_output()`: parses the text-based triple output format
//! - Integration with existing `ExtractedRelationship` type for pipeline compatibility
//!
//! Research basis: HippoRAG (NeurIPS 2024) demonstrates +20% Recall@5 from
//! KG-based retrieval. Triplex claims GPT-4-comparable extraction quality
//! at 1/60th the cost (SciPhi, July 2024).

use crate::error::{Error, Result};
use crate::extraction::output::ExtractedRelationship;
use crate::extraction::prompt::{
    apply_chat_template, build_triplex_extraction_prompt, ModelFamily,
};
use crate::inference::InferenceEngine;
use std::path::{Path, PathBuf};

/// Maximum tokens for Triplex generation.
/// Triples are short (~20 tokens each), but complex texts may produce 10-20 triples.
const TRIPLEX_MAX_TOKENS: u32 = 512;

/// KG triple extractor using SciPhi Triplex model.
///
/// Separate from `LlmEntityExtractor` because:
/// 1. Different model (Triplex vs Phi-4) with different prompt format
/// 2. Different output format (text triples vs JSON)
/// 3. Different purpose (KG construction vs entity/fact extraction)
///
/// Both can run simultaneously on 8GB+ VRAM (Full tier).
pub struct TriplexExtractor {
    engine: InferenceEngine,
    family: ModelFamily,
    model_path: PathBuf,
}

impl TriplexExtractor {
    /// Load Triplex model from a GGUF file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the Triplex GGUF file
    ///
    /// # Example
    /// ```rust,ignore
    /// let extractor = TriplexExtractor::load("models/triplex/Triplex-Q4_K_M.gguf")?;
    /// let triples = extractor.extract("Alice and Bob are friends", None)?;
    /// ```
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let family = ModelFamily::from_path(&model_path);
        let gpu_layers = InferenceEngine::detect_gpu_layers();

        let engine = InferenceEngine::load(&model_path, gpu_layers)?;

        Ok(Self {
            engine,
            family,
            model_path,
        })
    }

    /// Load Triplex model with explicit GPU layer count.
    ///
    /// Use `gpu_layers = 0` for CPU-only inference.
    pub fn load_with_gpu_layers(model_path: impl AsRef<Path>, gpu_layers: u32) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let family = ModelFamily::from_path(&model_path);

        let engine = InferenceEngine::load(&model_path, gpu_layers)?;

        Ok(Self {
            engine,
            family,
            model_path,
        })
    }

    /// Extract KG triples from text content.
    ///
    /// Returns `ExtractedRelationship` structs compatible with the existing
    /// `IngestionPipeline::store_relationships()` method.
    ///
    /// # Arguments
    /// * `content` - The text to extract triples from
    /// * `speaker` - Optional speaker name for attribution
    pub fn extract(
        &self,
        content: &str,
        speaker: Option<&str>,
    ) -> Result<Vec<ExtractedRelationship>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        let chatml_prompt = build_triplex_extraction_prompt(content, speaker);
        let prompt = apply_chat_template(&chatml_prompt, self.family);

        let raw_output = self.engine.generate(&prompt, TRIPLEX_MAX_TOKENS)?;

        Ok(parse_triplex_output(&raw_output))
    }

    /// Get the model path for diagnostics.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Reset the GPU context (for memory management during long ingestion runs).
    pub fn reset_context(&self) {
        self.engine.reset_context();
    }
}

/// A parsed triple from Triplex output.
#[derive(Debug, Clone, PartialEq)]
struct ParsedTriple {
    subject_type: String,
    subject_name: String,
    predicate: String,
    object_type: String,
    object_name: String,
}

/// Parse Triplex model output into `ExtractedRelationship` structs.
///
/// Handles three output formats:
///
/// **Format 1 — JSON `entities_and_triples` (most common):**
/// ```text
/// {"entities_and_triples": ["[1], PERSON:Alice", "[2], PERSON:Bob", "[1] FRIEND_OF [2]"]}
/// ```
/// May be wrapped in ```json ... ``` code blocks.
///
/// **Format 2 — Direct triples:**
/// ```text
/// PERSON:Alice > FRIEND_OF > PERSON:Bob
/// ```
///
/// **Format 3 — Numbered entity references (plain text):**
/// ```text
/// [1], PERSON:Alice
/// [2], PERSON:Bob
/// [1] FRIEND_OF [2]
/// ```
///
/// Filters to entity-to-entity relationships only (both sides must be named entities,
/// not dates/numbers/activities).
pub fn parse_triplex_output(output: &str) -> Vec<ExtractedRelationship> {
    // Try JSON format first (most common Triplex output)
    let relationships = parse_json_entities_and_triples(output);
    if !relationships.is_empty() {
        return relationships;
    }

    // Then try direct triple format
    let relationships = parse_direct_triples(output);
    if !relationships.is_empty() {
        return relationships;
    }

    // Finally try plain-text numbered references
    parse_numbered_references(output)
}

/// Parse JSON `entities_and_triples` format.
///
/// Triplex commonly outputs:
/// ```json
/// {"entities_and_triples": ["[1], PERSON:Alice", "[2], PERSON:Bob", "[1] FRIEND_OF [2]"]}
/// ```
/// May be wrapped in ```json ... ``` code blocks.
fn parse_json_entities_and_triples(output: &str) -> Vec<ExtractedRelationship> {
    // Strip code block markers
    let cleaned = output
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Try to parse as JSON
    #[derive(serde::Deserialize)]
    struct TriplexJson {
        entities_and_triples: Vec<String>,
    }

    let parsed: TriplexJson = match serde_json::from_str(cleaned) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };

    // Join all strings and parse as numbered references
    let joined = parsed.entities_and_triples.join("\n");
    parse_numbered_references(&joined)
}

/// Parse direct `TYPE:Name > PREDICATE > TYPE:Name` triples.
fn parse_direct_triples(output: &str) -> Vec<ExtractedRelationship> {
    let mut results = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Split on " > " to get [subject, predicate, object]
        let parts: Vec<&str> = line.split(" > ").collect();
        if parts.len() != 3 {
            continue;
        }

        if let (Some(subject), Some(object)) =
            (parse_typed_entity(parts[0]), parse_typed_entity(parts[2]))
        {
            let predicate = parts[1].trim().to_string();

            // Only keep entity-to-entity relationships (skip DATE, NUMBER, etc.)
            if is_entity_type(&subject.0) && is_entity_type(&object.0) {
                results.push(ExtractedRelationship {
                    from_entity: subject.1,
                    to_entity: object.1,
                    relation_type: normalize_predicate(&predicate),
                    confidence: 0.85, // Triplex doesn't output confidence scores
                });
            }
        }
    }

    results
}

/// Parse numbered reference format:
/// `[1], PERSON:Alice` + `[1] FRIEND_OF [2]`
fn parse_numbered_references(output: &str) -> Vec<ExtractedRelationship> {
    use std::collections::HashMap;

    let mut entities: HashMap<String, (String, String)> = HashMap::new(); // ref_id -> (type, name)
    let mut triples: Vec<(String, String, String)> = Vec::new(); // (from_ref, predicate, to_ref)

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Entity declaration: "[1], PERSON:Alice" or "[1], PERSON: Alice"
        if let Some(rest) = line.strip_prefix('[') {
            if let Some(bracket_end) = rest.find(']') {
                let ref_id = rest[..bracket_end].to_string();
                let after_bracket = rest[bracket_end + 1..].trim();

                if let Some(entity_part) = after_bracket.strip_prefix(',') {
                    // Entity declaration
                    if let Some((etype, name)) = parse_typed_entity(entity_part.trim()) {
                        entities.insert(ref_id, (etype, name));
                    }
                } else {
                    // Triple reference: "[1] FRIEND_OF [2]"
                    let parts: Vec<&str> = after_bracket.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        let predicate = parts[0].trim();
                        let obj_ref = parts[1]
                            .trim()
                            .trim_start_matches('[')
                            .trim_end_matches(']')
                            .to_string();
                        triples.push((ref_id, predicate.to_string(), obj_ref));
                    }
                }
            }
        }
    }

    // Resolve references to entities
    let mut results = Vec::new();
    for (from_ref, predicate, to_ref) in triples {
        if let (Some(from_entity), Some(to_entity)) =
            (entities.get(&from_ref), entities.get(&to_ref))
        {
            if is_entity_type(&from_entity.0) && is_entity_type(&to_entity.0) {
                results.push(ExtractedRelationship {
                    from_entity: from_entity.1.clone(),
                    to_entity: to_entity.1.clone(),
                    relation_type: normalize_predicate(&predicate),
                    confidence: 0.85,
                });
            }
        }
    }

    results
}

/// Parse a `TYPE:Name` or `TYPE: Name` string into (type, name).
fn parse_typed_entity(s: &str) -> Option<(String, String)> {
    let s = s.trim();
    let colon_pos = s.find(':')?;
    let etype = s[..colon_pos].trim().to_uppercase();
    let name = s[colon_pos + 1..].trim().to_string();
    if name.is_empty() {
        return None;
    }
    Some((etype, name))
}

/// Check if a Triplex entity type represents a named entity (not an abstract concept).
///
/// Only named entity types pass — these form meaningful KG nodes.
/// Abstract types like ACTIVITY ("camping"), CONCEPT ("husband"), DATE, NUMBER
/// are filtered because they produce noisy relationships.
fn is_entity_type(etype: &str) -> bool {
    matches!(
        etype,
        "PERSON"
            | "ORGANIZATION"
            | "LOCATION"
            | "ARTIST"
            | "CITY"
            | "COUNTRY"
            | "COMPANY"
            | "GROUP"
            | "TEAM"
    )
}

/// Normalize Triplex UPPER_CASE predicates to lowercase relationship types.
///
/// Converts `FRIEND_OF` → `friend`, `WORKS_AT` → `works at`, etc.
/// Strips trailing `_OF` for cleaner relationship labels.
fn normalize_predicate(predicate: &str) -> String {
    let lower = predicate.to_lowercase();

    // Strip common suffixes for cleaner labels
    let cleaned = if let Some(base) = lower.strip_suffix("_of") {
        base.to_string()
    } else {
        lower
    };

    // Replace underscores with spaces
    cleaned.replace('_', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_entities_and_triples() {
        let output = r#"```json
{
    "entities_and_triples": [
        "[1], PERSON:Melanie",
        "[2], PERSON:Dave",
        "[1] SPOUSE_OF [2]",
        "[3], ORGANIZATION:Google",
        "[2] WORKS_AT [3]",
        "[4], PERSON:Bob",
        "[1] FRIEND_OF [4]",
        "[2] COLLEAGUE_OF [4]"
    ]
}
```"#;
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 4);

        assert_eq!(results[0].from_entity, "Melanie");
        assert_eq!(results[0].to_entity, "Dave");
        assert_eq!(results[0].relation_type, "spouse");

        assert_eq!(results[1].from_entity, "Dave");
        assert_eq!(results[1].to_entity, "Google");
        assert_eq!(results[1].relation_type, "works at");

        assert_eq!(results[2].from_entity, "Melanie");
        assert_eq!(results[2].to_entity, "Bob");
        assert_eq!(results[2].relation_type, "friend");

        assert_eq!(results[3].from_entity, "Dave");
        assert_eq!(results[3].to_entity, "Bob");
        assert_eq!(results[3].relation_type, "colleague");
    }

    #[test]
    fn test_parse_json_without_code_blocks() {
        let output = r#"{"entities_and_triples": ["[1], PERSON:Alice", "[2], PERSON:Bob", "[1] FRIEND_OF [2]"]}"#;
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].from_entity, "Alice");
        assert_eq!(results[0].to_entity, "Bob");
    }

    #[test]
    fn test_parse_json_filters_non_entity_types() {
        let output = r#"{"entities_and_triples": ["[1], PERSON:Alice", "[2], ACTIVITY:camping", "[3], CONCEPT:husband", "[1] PARTICIPATES_IN [2]", "[1] FRIEND_OF [3]"]}"#;
        let results = parse_triplex_output(output);
        // ACTIVITY and CONCEPT are not entity types for relationships
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_parse_direct_triples_basic() {
        let output =
            "PERSON:Alice > FRIEND_OF > PERSON:Bob\nPERSON:Alice > WORKS_AT > ORGANIZATION:Google";
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 2);

        assert_eq!(results[0].from_entity, "Alice");
        assert_eq!(results[0].to_entity, "Bob");
        assert_eq!(results[0].relation_type, "friend");

        assert_eq!(results[1].from_entity, "Alice");
        assert_eq!(results[1].to_entity, "Google");
        assert_eq!(results[1].relation_type, "works at");
    }

    #[test]
    fn test_parse_direct_triples_with_spaces() {
        let output = "PERSON: Alice > SPOUSE_OF > PERSON: Bob";
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].from_entity, "Alice");
        assert_eq!(results[0].to_entity, "Bob");
        assert_eq!(results[0].relation_type, "spouse");
    }

    #[test]
    fn test_parse_direct_triples_filters_non_entities() {
        let output =
            "PERSON:Alice > BORN_ON > DATE:1990-01-15\nPERSON:Alice > FRIEND_OF > PERSON:Bob";
        let results = parse_triplex_output(output);
        // DATE is not an entity type, so first triple is filtered
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].to_entity, "Bob");
    }

    #[test]
    fn test_parse_numbered_references() {
        let output = "[1], PERSON:Alice\n[2], PERSON:Bob\n[3], ORGANIZATION:Google\n[1] FRIEND_OF [2]\n[1] WORKS_AT [3]";
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 2);

        assert_eq!(results[0].from_entity, "Alice");
        assert_eq!(results[0].to_entity, "Bob");
        assert_eq!(results[0].relation_type, "friend");

        assert_eq!(results[1].from_entity, "Alice");
        assert_eq!(results[1].to_entity, "Google");
        assert_eq!(results[1].relation_type, "works at");
    }

    #[test]
    fn test_parse_numbered_references_filters_non_entities() {
        let output = "[1], PERSON:Alice\n[2], DATE:2022\n[1] BORN_ON [2]";
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_parse_empty_output() {
        let results = parse_triplex_output("");
        assert!(results.is_empty());

        let results = parse_triplex_output("\n\n\n");
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_malformed_lines_ignored() {
        let output =
            "This is not a triple\nPERSON:Alice > FRIEND_OF > PERSON:Bob\nAnother garbage line";
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].from_entity, "Alice");
    }

    #[test]
    fn test_normalize_predicate() {
        assert_eq!(normalize_predicate("FRIEND_OF"), "friend");
        assert_eq!(normalize_predicate("SPOUSE_OF"), "spouse");
        assert_eq!(normalize_predicate("WORKS_AT"), "works at");
        assert_eq!(normalize_predicate("LIVES_IN"), "lives in");
        assert_eq!(normalize_predicate("COLLEAGUE_OF"), "colleague");
        assert_eq!(normalize_predicate("INTERESTED_IN"), "interested in");
        assert_eq!(normalize_predicate("KNOWS"), "knows");
    }

    #[test]
    fn test_parse_typed_entity() {
        assert_eq!(
            parse_typed_entity("PERSON:Alice"),
            Some(("PERSON".to_string(), "Alice".to_string()))
        );
        assert_eq!(
            parse_typed_entity("PERSON: Alice Smith"),
            Some(("PERSON".to_string(), "Alice Smith".to_string()))
        );
        assert_eq!(
            parse_typed_entity("ORGANIZATION:Google Inc."),
            Some(("ORGANIZATION".to_string(), "Google Inc.".to_string()))
        );
        assert_eq!(parse_typed_entity("PERSON:"), None);
        assert_eq!(parse_typed_entity("no colon"), None);
    }

    #[test]
    fn test_is_entity_type() {
        assert!(is_entity_type("PERSON"));
        assert!(is_entity_type("ORGANIZATION"));
        assert!(is_entity_type("LOCATION"));
        assert!(is_entity_type("CITY"));
        assert!(is_entity_type("COUNTRY"));
        // Abstract types filtered — not named entities
        assert!(!is_entity_type("ACTIVITY"));
        assert!(!is_entity_type("CONCEPT"));
        assert!(!is_entity_type("OBJECT"));
        assert!(!is_entity_type("DATE"));
        assert!(!is_entity_type("NUMBER"));
        assert!(!is_entity_type("POSITION"));
    }

    #[test]
    fn test_confidence_default() {
        let output = "PERSON:Alice > FRIEND_OF > PERSON:Bob";
        let results = parse_triplex_output(output);
        assert_eq!(results[0].confidence, 0.85);
    }

    #[test]
    fn test_complex_names() {
        // CONCEPT is filtered (not a named entity), so this produces 0 results
        let output = "PERSON:Vincent van Gogh > BELONGS_TO_MOVEMENT > CONCEPT:post-impressionism";
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 0);

        // Person-to-Organization should work with complex names
        let output = "PERSON:Vincent van Gogh > WORKS_AT > ORGANIZATION:Atelier de Cormon";
        let results = parse_triplex_output(output);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].from_entity, "Vincent van Gogh");
        assert_eq!(results[0].to_entity, "Atelier de Cormon");
    }

    #[test]
    fn test_mixed_format_prefers_direct() {
        // If both formats appear, direct triples take precedence
        let output = "PERSON:Alice > FRIEND_OF > PERSON:Bob\n[1], PERSON:Carol\n[2], PERSON:Dave\n[1] KNOWS [2]";
        let results = parse_triplex_output(output);
        // Direct format finds at least 1 triple, so numbered refs are skipped
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].from_entity, "Alice");
    }
}
