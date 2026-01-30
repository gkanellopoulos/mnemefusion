//! Unit tests for SLM integration
//!
//! Tests configuration, initialization, fallback behavior, and error handling.
//! Does NOT test actual model inference (requires downloading 5GB model).

#[cfg(test)]
mod tests {
    use crate::slm::{SlmClassifier, SlmConfig};
    use crate::query::intent::QueryIntent;

    #[test]
    fn test_slm_config_default() {
        let config = SlmConfig::default();

        assert_eq!(config.model_id, "google/gemma-2-2b-it");
        assert_eq!(config.timeout_ms, 100);
        assert!(!config.use_gpu);
        assert_eq!(config.min_confidence, 0.6);
        assert!(config.cache_dir.to_string_lossy().contains("mnemefusion"));
        assert!(config.model_path.is_none()); // No local path by default
    }

    #[test]
    fn test_slm_config_builder() {
        let config = SlmConfig::new("custom/model")
            .with_timeout_ms(500)
            .with_gpu(true)
            .with_min_confidence(0.8);

        assert_eq!(config.model_id, "custom/model");
        assert_eq!(config.timeout_ms, 500);
        assert!(config.use_gpu);
        assert_eq!(config.min_confidence, 0.8);
    }

    #[test]
    fn test_slm_config_cache_dir() {
        let custom_dir = std::path::PathBuf::from("/tmp/test_cache");
        let config = SlmConfig::default()
            .with_cache_dir(custom_dir.clone());

        assert_eq!(config.cache_dir, custom_dir);
    }

    #[test]
    fn test_slm_config_model_path() {
        let model_path = std::path::PathBuf::from("/models/gemma-2-2b-it");
        let config = SlmConfig::default()
            .with_model_path(model_path.clone());

        assert_eq!(config.model_path, Some(model_path));
    }

    #[cfg(feature = "slm")]
    #[test]
    fn test_slm_classifier_creation() {
        let config = SlmConfig::default();
        let classifier = SlmClassifier::new(config);

        // Should create successfully (model not loaded yet)
        assert!(classifier.is_ok());
    }

    #[cfg(feature = "slm")]
    #[test]
    fn test_slm_classifier_lazy_loading() {
        let config = SlmConfig::default();
        let classifier = SlmClassifier::new(config).unwrap();

        // Classifier created but model not loaded yet
        // (Model loads on first classify_intent() call)
        // No way to check internal state, but creation should be fast
    }

    #[cfg(not(feature = "slm"))]
    #[test]
    fn test_slm_disabled_returns_error() {
        let config = SlmConfig::default();
        let classifier = SlmClassifier::new(config);

        // Should return error when SLM feature disabled
        assert!(classifier.is_err());
        assert!(matches!(classifier.unwrap_err(), crate::Error::SlmNotAvailable));
    }

    #[test]
    fn test_parse_entity_output() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let json = r#"{
                "intent": "Entity",
                "confidence": 0.95,
                "entity_focus": "Sarah",
                "reasoning": "Query asks about a person"
            }"#;

            let result = classifier.parse_intent_output(json).unwrap();
            assert_eq!(result.intent, QueryIntent::Entity);
            assert_eq!(result.confidence, 0.95);
            assert_eq!(result.entity_focus, Some("Sarah".to_string()));
        }
    }

    #[test]
    fn test_parse_temporal_output() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let json = r#"{
                "intent": "Temporal",
                "confidence": 0.88,
                "entity_focus": null,
                "reasoning": "Query asks about time"
            }"#;

            let result = classifier.parse_intent_output(json).unwrap();
            assert_eq!(result.intent, QueryIntent::Temporal);
            assert_eq!(result.confidence, 0.88);
            assert_eq!(result.entity_focus, None);
        }
    }

    #[test]
    fn test_parse_causal_output() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let json = r#"{
                "intent": "Causal",
                "confidence": 0.92,
                "entity_focus": null,
                "reasoning": "Query asks why"
            }"#;

            let result = classifier.parse_intent_output(json).unwrap();
            assert_eq!(result.intent, QueryIntent::Causal);
            assert_eq!(result.confidence, 0.92);
            assert_eq!(result.entity_focus, None);
        }
    }

    #[test]
    fn test_parse_factual_output() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let json = r#"{
                "intent": "Factual",
                "confidence": 0.75,
                "entity_focus": null,
                "reasoning": "Generic query"
            }"#;

            let result = classifier.parse_intent_output(json).unwrap();
            assert_eq!(result.intent, QueryIntent::Factual);
            assert_eq!(result.confidence, 0.75);
            assert_eq!(result.entity_focus, None);
        }
    }

    #[test]
    fn test_parse_output_with_extra_text() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let output = r#"Sure, let me help classify that.
            {
                "intent": "Entity",
                "confidence": 0.90,
                "entity_focus": "project",
                "reasoning": "Query about entity"
            }
            Hope this helps!"#;

            let result = classifier.parse_intent_output(output).unwrap();
            assert_eq!(result.intent, QueryIntent::Entity);
            assert_eq!(result.confidence, 0.90);
        }
    }

    #[test]
    fn test_parse_invalid_json() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let invalid_json = "This is not JSON";
            let result = classifier.parse_intent_output(invalid_json);

            assert!(result.is_err());
        }
    }

    #[test]
    fn test_parse_incomplete_json() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let incomplete = r#"{"intent": "Entity""#;
            let result = classifier.parse_intent_output(incomplete);

            assert!(result.is_err());
        }
    }

    #[test]
    fn test_parse_missing_intent_field() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let missing_intent = r#"{
                "confidence": 0.9,
                "entity_focus": null
            }"#;

            let result = classifier.parse_intent_output(missing_intent);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_parse_unknown_intent() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let unknown = r#"{
                "intent": "UnknownIntent",
                "confidence": 0.9,
                "entity_focus": null
            }"#;

            // Should default to Factual for unknown intents
            let result = classifier.parse_intent_output(unknown).unwrap();
            assert_eq!(result.intent, QueryIntent::Factual);
        }
    }

    #[test]
    fn test_prompt_generation() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            let prompt = classifier.create_classification_prompt("Who is Sarah?");

            assert!(prompt.contains("Who is Sarah?"));
            assert!(prompt.contains("Entity|Temporal|Causal|Factual"));
            assert!(prompt.contains("JSON"));
            assert!(prompt.contains("<start_of_turn>"));
            assert!(prompt.contains("<end_of_turn>"));
        }
    }

    #[test]
    fn test_confidence_bounds() {
        #[cfg(feature = "slm")]
        {
            let config = SlmConfig::default();
            let classifier = SlmClassifier::new(config).unwrap();

            // Test confidence at boundaries
            let json = r#"{
                "intent": "Entity",
                "confidence": 1.0,
                "entity_focus": null
            }"#;

            let result = classifier.parse_intent_output(json).unwrap();
            assert_eq!(result.confidence, 1.0);

            // Test zero confidence (should still parse)
            let json_zero = r#"{
                "intent": "Temporal",
                "confidence": 0.0,
                "entity_focus": null
            }"#;

            let result_zero = classifier.parse_intent_output(json_zero).unwrap();
            assert_eq!(result_zero.confidence, 0.0);
        }
    }
}
