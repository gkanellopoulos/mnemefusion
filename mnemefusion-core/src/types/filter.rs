use std::fmt;

/// Filter operator for metadata field comparisons
#[derive(Debug, Clone, PartialEq)]
pub enum FilterOp {
    /// Exact match: field == value
    Eq(String),

    /// Greater than: field > value
    Gt(String),

    /// Greater than or equal: field >= value
    Gte(String),

    /// Less than: field < value
    Lt(String),

    /// Less than or equal: field <= value
    Lte(String),

    /// In list: field in [values]
    In(Vec<String>),

    /// Not equal: field != value
    Ne(String),
}

impl FilterOp {
    /// Evaluate the filter operation against a value
    pub fn matches(&self, value: &str) -> bool {
        match self {
            FilterOp::Eq(target) => value == target,
            FilterOp::Ne(target) => value != target,
            FilterOp::Gt(target) => value > target.as_str(),
            FilterOp::Gte(target) => value >= target.as_str(),
            FilterOp::Lt(target) => value < target.as_str(),
            FilterOp::Lte(target) => value <= target.as_str(),
            FilterOp::In(targets) => targets.iter().any(|t| t == value),
        }
    }
}

impl fmt::Display for FilterOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterOp::Eq(v) => write!(f, "= {}", v),
            FilterOp::Ne(v) => write!(f, "!= {}", v),
            FilterOp::Gt(v) => write!(f, "> {}", v),
            FilterOp::Gte(v) => write!(f, ">= {}", v),
            FilterOp::Lt(v) => write!(f, "< {}", v),
            FilterOp::Lte(v) => write!(f, "<= {}", v),
            FilterOp::In(v) => write!(f, "IN {:?}", v),
        }
    }
}

/// A metadata filter specification
#[derive(Debug, Clone)]
pub struct MetadataFilter {
    /// The metadata field to filter on
    pub field: String,

    /// The filter operation to apply
    pub op: FilterOp,
}

impl MetadataFilter {
    /// Create a new metadata filter
    pub fn new(field: impl Into<String>, op: FilterOp) -> Self {
        Self {
            field: field.into(),
            op,
        }
    }

    /// Create an exact match filter
    pub fn eq(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(field, FilterOp::Eq(value.into()))
    }

    /// Create a not-equal filter
    pub fn ne(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(field, FilterOp::Ne(value.into()))
    }

    /// Create a greater-than filter
    pub fn gt(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(field, FilterOp::Gt(value.into()))
    }

    /// Create a greater-than-or-equal filter
    pub fn gte(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(field, FilterOp::Gte(value.into()))
    }

    /// Create a less-than filter
    pub fn lt(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(field, FilterOp::Lt(value.into()))
    }

    /// Create a less-than-or-equal filter
    pub fn lte(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(field, FilterOp::Lte(value.into()))
    }

    /// Create an in-list filter
    pub fn in_list(field: impl Into<String>, values: Vec<String>) -> Self {
        Self::new(field, FilterOp::In(values))
    }

    /// Check if a metadata value matches this filter
    pub fn matches(&self, value: Option<&str>) -> bool {
        match value {
            Some(v) => self.op.matches(v),
            None => false, // Missing fields don't match any filter
        }
    }
}

impl fmt::Display for MetadataFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.field, self.op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_op_eq() {
        let op = FilterOp::Eq("test".to_string());
        assert!(op.matches("test"));
        assert!(!op.matches("other"));
    }

    #[test]
    fn test_filter_op_ne() {
        let op = FilterOp::Ne("test".to_string());
        assert!(!op.matches("test"));
        assert!(op.matches("other"));
    }

    #[test]
    fn test_filter_op_gt() {
        let op = FilterOp::Gt("5".to_string());
        assert!(op.matches("6"));
        assert!(op.matches("50"));
        assert!(!op.matches("5"));
        assert!(!op.matches("4"));
    }

    #[test]
    fn test_filter_op_gte() {
        let op = FilterOp::Gte("5".to_string());
        assert!(op.matches("6"));
        assert!(op.matches("5"));
        assert!(!op.matches("4"));
    }

    #[test]
    fn test_filter_op_lt() {
        let op = FilterOp::Lt("5".to_string());
        assert!(op.matches("4"));
        assert!(op.matches("10")); // String comparison: "10" < "5"
        assert!(!op.matches("5"));
        assert!(!op.matches("6"));
    }

    #[test]
    fn test_filter_op_lte() {
        let op = FilterOp::Lte("5".to_string());
        assert!(op.matches("4"));
        assert!(op.matches("5"));
        assert!(!op.matches("6"));
    }

    #[test]
    fn test_filter_op_in() {
        let op = FilterOp::In(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert!(op.matches("a"));
        assert!(op.matches("b"));
        assert!(op.matches("c"));
        assert!(!op.matches("d"));
    }

    #[test]
    fn test_metadata_filter_eq() {
        let filter = MetadataFilter::eq("type", "event");
        assert!(filter.matches(Some("event")));
        assert!(!filter.matches(Some("task")));
        assert!(!filter.matches(None));
    }

    #[test]
    fn test_metadata_filter_ne() {
        let filter = MetadataFilter::ne("status", "archived");
        assert!(filter.matches(Some("active")));
        assert!(!filter.matches(Some("archived")));
        assert!(!filter.matches(None));
    }

    #[test]
    fn test_metadata_filter_gte() {
        let filter = MetadataFilter::gte("priority", "5");
        assert!(filter.matches(Some("5")));
        assert!(filter.matches(Some("6")));
        assert!(!filter.matches(Some("4")));
        assert!(!filter.matches(None));
    }

    #[test]
    fn test_metadata_filter_in() {
        let filter =
            MetadataFilter::in_list("category", vec!["food".to_string(), "travel".to_string()]);
        assert!(filter.matches(Some("food")));
        assert!(filter.matches(Some("travel")));
        assert!(!filter.matches(Some("work")));
        assert!(!filter.matches(None));
    }

    #[test]
    fn test_filter_display() {
        let filter = MetadataFilter::eq("type", "event");
        assert_eq!(format!("{}", filter), "type = event");

        let filter = MetadataFilter::gte("count", "10");
        assert_eq!(format!("{}", filter), "count >= 10");
    }
}
