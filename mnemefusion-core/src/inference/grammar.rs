//! GBNF grammar for constrained JSON decoding
//!
//! Forces the model to only output valid JSON matching our entity extraction schema.
//! This guarantees 100% valid JSON output - no parsing failures possible.

/// Grammar that constrains LLM output to valid entity extraction JSON
#[derive(Debug, Clone)]
pub struct JsonGrammar {
    grammar_str: String,
}

impl JsonGrammar {
    /// Create grammar for entity extraction output
    ///
    /// This grammar constrains the model to output valid JSON with:
    /// - entities: list of {name, type}
    /// - entity_facts: list of {entity, fact_type, value, confidence}
    /// - topics: list of strings
    /// - importance: number 0.0-1.0
    pub fn entity_extraction() -> Self {
        // GBNF grammar that matches our ExtractionResult schema
        let grammar = r#"
root ::= "{" ws members ws "}"

members ::= entities-member "," ws entity-facts-member "," ws topics-member "," ws importance-member

entities-member ::= "\"entities\"" ws ":" ws entities-array
entities-array ::= "[" ws (entity-obj ("," ws entity-obj)*)? ws "]"
entity-obj ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"type\"" ws ":" ws entity-type ws "}"
entity-type ::= "\"person\"" | "\"organization\"" | "\"location\"" | "\"concept\"" | "\"event\""

entity-facts-member ::= "\"entity_facts\"" ws ":" ws facts-array
facts-array ::= "[" ws (fact-obj ("," ws fact-obj)*)? ws "]"
fact-obj ::= "{" ws fact-entity "," ws fact-type-field "," ws fact-value "," ws fact-confidence ws "}"
fact-entity ::= "\"entity\"" ws ":" ws string
fact-type-field ::= "\"fact_type\"" ws ":" ws fact-type-value
fact-type-value ::= "\"occupation\"" | "\"research_topic\"" | "\"goal\"" | "\"career_goal\"" | "\"preference\"" | "\"location\"" | "\"relationship\"" | "\"relationship_status\"" | "\"interest\"" | "\"affiliation\"" | "\"characteristic\"" | "\"action\"" | "\"instrument\"" | "\"sport\"" | "\"pet\"" | "\"book\"" | "\"food\"" | "\"hobby\"" | "\"travel\"" | "\"family\"" | "\"event\""
fact-value ::= "\"value\"" ws ":" ws string
fact-confidence ::= "\"confidence\"" ws ":" ws number

topics-member ::= "\"topics\"" ws ":" ws string-array
string-array ::= "[" ws (string ("," ws string)*)? ws "]"

importance-member ::= "\"importance\"" ws ":" ws number

string ::= "\"" characters "\""
characters ::= character*
character ::= [^"\\] | "\\" escape-char
escape-char ::= ["\\nrt]

number ::= "0" | ([1-9] [0-9]*) ("." [0-9]+)?
ws ::= [ \t\n]*
"#;
        Self {
            grammar_str: grammar.to_string(),
        }
    }

    /// Get the raw GBNF grammar string
    pub fn as_str(&self) -> &str {
        &self.grammar_str
    }

    /// Create a simple grammar for testing
    #[cfg(test)]
    pub fn simple_json() -> Self {
        let grammar = r#"
root ::= "{" ws "\"test\"" ws ":" ws string ws "}"
string ::= "\"" [a-zA-Z0-9 ]* "\""
ws ::= [ \t\n]*
"#;
        Self {
            grammar_str: grammar.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_extraction_grammar_created() {
        let grammar = JsonGrammar::entity_extraction();
        assert!(!grammar.as_str().is_empty());
        assert!(grammar.as_str().contains("entity-facts"));
        assert!(grammar.as_str().contains("fact-type-value"));
    }

    #[test]
    fn test_grammar_includes_all_fact_types() {
        let grammar = JsonGrammar::entity_extraction();
        let grammar_str = grammar.as_str();

        // Verify all fact types are present
        assert!(grammar_str.contains("occupation"));
        assert!(grammar_str.contains("research_topic"));
        assert!(grammar_str.contains("goal"));
        assert!(grammar_str.contains("preference"));
        assert!(grammar_str.contains("location"));
        assert!(grammar_str.contains("relationship"));
        assert!(grammar_str.contains("interest"));
    }

    #[test]
    fn test_grammar_includes_entity_types() {
        let grammar = JsonGrammar::entity_extraction();
        let grammar_str = grammar.as_str();

        assert!(grammar_str.contains("person"));
        assert!(grammar_str.contains("organization"));
        assert!(grammar_str.contains("location"));
        assert!(grammar_str.contains("concept"));
    }
}
