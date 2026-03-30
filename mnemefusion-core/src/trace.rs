//! Pipeline observability / tracing
//!
//! Provides structured step-by-step traces of query and ingestion pipelines.
//! Gated by `Config::enable_trace` — zero overhead when disabled.
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut rec = TraceRecorder::new("query");
//! rec.begin_step("semantic_search");
//! rec.record("candidate_count", 140_i64);
//! rec.record("top_5_scores", vec![0.82f32, 0.71, 0.68, 0.65, 0.61]);
//! rec.end_step();
//! let trace = rec.finish();
//! ```

use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Public output types (immutable, serializable)
// ---------------------------------------------------------------------------

/// Immutable trace output — serializable snapshot of a pipeline execution.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Trace {
    pub pipeline: String,
    pub total_duration_us: u64,
    pub steps: Vec<TraceStep>,
}

/// One step within a pipeline trace.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TraceStep {
    pub name: String,
    pub duration_us: u64,
    pub data: HashMap<String, TraceValue>,
}

/// Heterogeneous value type for trace data.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum TraceValue {
    String(String),
    Float(f64),
    Int(i64),
    Bool(bool),
    List(Vec<TraceValue>),
    Map(HashMap<String, TraceValue>),
    Null,
}

// ---------------------------------------------------------------------------
// Into<TraceValue> implementations
// ---------------------------------------------------------------------------

impl From<&str> for TraceValue {
    fn from(v: &str) -> Self {
        TraceValue::String(v.to_string())
    }
}

impl From<String> for TraceValue {
    fn from(v: String) -> Self {
        TraceValue::String(v)
    }
}

impl From<f32> for TraceValue {
    fn from(v: f32) -> Self {
        TraceValue::Float(v as f64)
    }
}

impl From<f64> for TraceValue {
    fn from(v: f64) -> Self {
        TraceValue::Float(v)
    }
}

impl From<i32> for TraceValue {
    fn from(v: i32) -> Self {
        TraceValue::Int(v as i64)
    }
}

impl From<i64> for TraceValue {
    fn from(v: i64) -> Self {
        TraceValue::Int(v)
    }
}

impl From<usize> for TraceValue {
    fn from(v: usize) -> Self {
        TraceValue::Int(v as i64)
    }
}

impl From<u32> for TraceValue {
    fn from(v: u32) -> Self {
        TraceValue::Int(v as i64)
    }
}

impl From<bool> for TraceValue {
    fn from(v: bool) -> Self {
        TraceValue::Bool(v)
    }
}

impl From<Vec<String>> for TraceValue {
    fn from(v: Vec<String>) -> Self {
        TraceValue::List(v.into_iter().map(TraceValue::String).collect())
    }
}

impl From<Vec<f32>> for TraceValue {
    fn from(v: Vec<f32>) -> Self {
        TraceValue::List(v.into_iter().map(|f| TraceValue::Float(f as f64)).collect())
    }
}

impl From<HashMap<String, f32>> for TraceValue {
    fn from(v: HashMap<String, f32>) -> Self {
        TraceValue::Map(
            v.into_iter()
                .map(|(k, f)| (k, TraceValue::Float(f as f64)))
                .collect(),
        )
    }
}

impl<T: Into<TraceValue>> From<Option<T>> for TraceValue {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(inner) => inner.into(),
            None => TraceValue::Null,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience macro
// ---------------------------------------------------------------------------

/// Record a key-value pair on an `Option<&mut TraceRecorder>`.
/// No-op when the recorder is `None`.
#[macro_export]
macro_rules! trace_record {
    ($recorder:expr, $key:expr, $value:expr) => {
        if let Some(ref mut __r) = $recorder {
            __r.record($key, $value);
        }
    };
}

/// Begin a named step on an `Option<&mut TraceRecorder>`.
/// No-op when the recorder is `None`.
#[macro_export]
macro_rules! trace_begin {
    ($recorder:expr, $name:expr) => {
        if let Some(ref mut __r) = $recorder {
            __r.begin_step($name);
        }
    };
}

// ---------------------------------------------------------------------------
// Mutable recorder (threaded through pipeline execution)
// ---------------------------------------------------------------------------

struct ActiveStep {
    name: String,
    started_at: Instant,
    data: HashMap<String, TraceValue>,
}

/// Mutable recorder threaded through pipeline execution.
///
/// Only allocated when `Config::enable_trace` is `true`. The recorder
/// collects steps with timing and arbitrary key-value data, then produces
/// an immutable [`Trace`] via [`finish()`](Self::finish).
pub struct TraceRecorder {
    pipeline: String,
    started_at: Instant,
    steps: Vec<TraceStep>,
    current_step: Option<ActiveStep>,
}

impl TraceRecorder {
    /// Create a new recorder for the given pipeline name (e.g., `"query"`, `"ingest"`).
    pub fn new(pipeline: &str) -> Self {
        Self {
            pipeline: pipeline.to_string(),
            started_at: Instant::now(),
            steps: Vec::new(),
            current_step: None,
        }
    }

    /// Begin a named step. Automatically ends the previous step if one is open.
    pub fn begin_step(&mut self, name: &str) {
        self.close_current_step();
        self.current_step = Some(ActiveStep {
            name: name.to_string(),
            started_at: Instant::now(),
            data: HashMap::new(),
        });
    }

    /// Record a key-value pair on the current step.
    ///
    /// Silently ignored if no step is open.
    pub fn record(&mut self, key: &str, value: impl Into<TraceValue>) {
        if let Some(ref mut step) = self.current_step {
            step.data.insert(key.to_string(), value.into());
        }
    }

    /// End the current step explicitly.
    pub fn end_step(&mut self) {
        self.close_current_step();
    }

    /// Finalize and return the immutable [`Trace`].
    ///
    /// Automatically closes any open step.
    pub fn finish(mut self) -> Trace {
        self.close_current_step();
        Trace {
            pipeline: self.pipeline,
            total_duration_us: self.started_at.elapsed().as_micros() as u64,
            steps: self.steps,
        }
    }

    fn close_current_step(&mut self) {
        if let Some(step) = self.current_step.take() {
            self.steps.push(TraceStep {
                name: step.name,
                duration_us: step.started_at.elapsed().as_micros() as u64,
                data: step.data,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recorder_basic_lifecycle() {
        let mut rec = TraceRecorder::new("query");
        rec.begin_step("step_a");
        rec.record("count", 42_i64);
        rec.end_step();

        let trace = rec.finish();
        assert_eq!(trace.pipeline, "query");
        assert_eq!(trace.steps.len(), 1);
        assert_eq!(trace.steps[0].name, "step_a");
        assert!(matches!(
            trace.steps[0].data.get("count"),
            Some(TraceValue::Int(42))
        ));
    }

    #[test]
    fn test_auto_close_on_begin_step() {
        let mut rec = TraceRecorder::new("query");
        rec.begin_step("step_a");
        rec.record("x", 1_i64);
        // begin_step auto-closes step_a
        rec.begin_step("step_b");
        rec.record("y", 2_i64);

        let trace = rec.finish();
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.steps[0].name, "step_a");
        assert_eq!(trace.steps[1].name, "step_b");
    }

    #[test]
    fn test_auto_close_on_finish() {
        let mut rec = TraceRecorder::new("ingest");
        rec.begin_step("step_a");
        rec.record("val", "hello");

        let trace = rec.finish();
        assert_eq!(trace.steps.len(), 1);
        assert!(matches!(
            trace.steps[0].data.get("val"),
            Some(TraceValue::String(s)) if s == "hello"
        ));
    }

    #[test]
    fn test_record_without_open_step_is_noop() {
        let mut rec = TraceRecorder::new("query");
        rec.record("orphan", 99_i64); // no step open — silently ignored
        let trace = rec.finish();
        assert_eq!(trace.steps.len(), 0);
    }

    #[test]
    fn test_all_value_types() {
        let mut rec = TraceRecorder::new("query");
        rec.begin_step("types");
        rec.record("str_val", "hello");
        rec.record("string_val", String::from("world"));
        rec.record("f32_val", 3.14f32);
        rec.record("f64_val", 2.718f64);
        rec.record("i32_val", -5i32);
        rec.record("i64_val", 100i64);
        rec.record("usize_val", 42usize);
        rec.record("bool_val", true);
        rec.record("vec_string", vec!["a".to_string(), "b".to_string()]);
        rec.record("vec_f32", vec![1.0f32, 2.0, 3.0]);
        rec.record("none_val", Option::<String>::None);
        rec.record("some_val", Some(7i64));

        let trace = rec.finish();
        let data = &trace.steps[0].data;

        assert!(matches!(data.get("str_val"), Some(TraceValue::String(s)) if s == "hello"));
        assert!(matches!(data.get("f32_val"), Some(TraceValue::Float(f)) if (*f - 3.14).abs() < 0.01));
        assert!(matches!(data.get("i32_val"), Some(TraceValue::Int(-5))));
        assert!(matches!(data.get("bool_val"), Some(TraceValue::Bool(true))));
        assert!(matches!(data.get("none_val"), Some(TraceValue::Null)));
        assert!(matches!(data.get("some_val"), Some(TraceValue::Int(7))));

        if let Some(TraceValue::List(items)) = data.get("vec_string") {
            assert_eq!(items.len(), 2);
        } else {
            panic!("vec_string should be a List");
        }

        if let Some(TraceValue::List(items)) = data.get("vec_f32") {
            assert_eq!(items.len(), 3);
        } else {
            panic!("vec_f32 should be a List");
        }
    }

    #[test]
    fn test_hashmap_value() {
        let mut rec = TraceRecorder::new("query");
        rec.begin_step("map_test");
        let mut map = HashMap::new();
        map.insert("semantic".to_string(), 0.8f32);
        map.insert("bm25".to_string(), 0.3f32);
        rec.record("scores", map);

        let trace = rec.finish();
        if let Some(TraceValue::Map(m)) = trace.steps[0].data.get("scores") {
            assert_eq!(m.len(), 2);
            assert!(matches!(m.get("semantic"), Some(TraceValue::Float(f)) if (*f - 0.8).abs() < 0.01));
        } else {
            panic!("scores should be a Map");
        }
    }

    #[test]
    fn test_trace_serializable() {
        let mut rec = TraceRecorder::new("query");
        rec.begin_step("step_a");
        rec.record("count", 10_i64);
        let trace = rec.finish();

        let json = serde_json::to_string(&trace).expect("Trace should be serializable");
        assert!(json.contains("\"pipeline\":\"query\""));
        assert!(json.contains("\"step_a\""));
    }

    #[test]
    fn test_duration_recorded() {
        let mut rec = TraceRecorder::new("query");
        rec.begin_step("step_a");
        // Tiny sleep to ensure non-zero duration
        std::thread::sleep(std::time::Duration::from_micros(100));
        rec.end_step();

        let trace = rec.finish();
        assert!(trace.steps[0].duration_us > 0);
        assert!(trace.total_duration_us > 0);
    }

    #[test]
    fn test_macro_with_some_recorder() {
        let mut recorder = Some(TraceRecorder::new("query"));
        trace_begin!(recorder, "macro_step");
        trace_record!(recorder, "key", 42_i64);

        let trace = recorder.unwrap().finish();
        assert_eq!(trace.steps.len(), 1);
        assert_eq!(trace.steps[0].name, "macro_step");
    }

    #[test]
    fn test_macro_with_none_recorder() {
        let mut recorder: Option<TraceRecorder> = None;
        // Should compile and be a no-op
        trace_begin!(recorder, "nothing");
        trace_record!(recorder, "key", 42_i64);
        assert!(recorder.is_none());
    }
}
