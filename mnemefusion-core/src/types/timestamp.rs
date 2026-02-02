//! Timestamp utilities for temporal indexing
//!
//! This module provides timestamp handling with microsecond precision.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Timestamp with microsecond precision
///
/// Used for temporal indexing and decay calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(u64); // microseconds since UNIX_EPOCH

impl Timestamp {
    /// Get current timestamp
    pub fn now() -> Self {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time before UNIX_EPOCH");
        Self(duration.as_micros() as u64)
    }

    /// Create timestamp from Unix seconds
    pub fn from_unix_secs(secs: f64) -> Self {
        Self((secs * 1_000_000.0) as u64)
    }

    /// Convert to Unix seconds
    pub fn as_unix_secs(&self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    /// Create timestamp from microseconds
    pub fn from_micros(micros: u64) -> Self {
        Self(micros)
    }

    /// Get microseconds since UNIX_EPOCH
    pub fn as_micros(&self) -> u64 {
        self.0
    }

    /// Subtract days from timestamp
    pub fn subtract_days(&self, days: u64) -> Self {
        let micros_per_day = 24 * 60 * 60 * 1_000_000u64;
        let to_subtract = days * micros_per_day;
        Self(self.0.saturating_sub(to_subtract))
    }

    /// Add days to timestamp
    pub fn add_days(&self, days: u64) -> Self {
        let micros_per_day = 24 * 60 * 60 * 1_000_000u64;
        Self(self.0 + days * micros_per_day)
    }

    /// Get start of day (midnight UTC)
    pub fn start_of_day(&self) -> Self {
        let micros_per_day = 24 * 60 * 60 * 1_000_000u64;
        let days_since_epoch = self.0 / micros_per_day;
        Self(days_since_epoch * micros_per_day)
    }

    /// Get end of day (23:59:59.999999 UTC)
    pub fn end_of_day(&self) -> Self {
        let micros_per_day = 24 * 60 * 60 * 1_000_000u64;
        let days_since_epoch = self.0 / micros_per_day;
        Self((days_since_epoch + 1) * micros_per_day - 1)
    }

    /// Calculate seconds between two timestamps
    pub fn seconds_since(&self, earlier: &Timestamp) -> f64 {
        if self.0 >= earlier.0 {
            (self.0 - earlier.0) as f64 / 1_000_000.0
        } else {
            0.0
        }
    }

    /// Calculate seconds until another timestamp
    pub fn seconds_until(&self, later: &Timestamp) -> f64 {
        later.seconds_since(self)
    }

    /// Check if timestamp is in the future
    pub fn is_future(&self) -> bool {
        *self > Self::now()
    }

    /// Check if timestamp is in the past
    pub fn is_past(&self) -> bool {
        *self < Self::now()
    }

    /// Convert to bytes for storage
    pub fn to_bytes(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 8 {
            return Err(Error::InvalidTimestamp(format!(
                "Expected 8 bytes, got {}",
                bytes.len()
            )));
        }
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(Self(u64::from_le_bytes(array)))
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::now()
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_unix_secs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_now() {
        let t1 = Timestamp::now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = Timestamp::now();
        assert!(t2 > t1);
    }

    #[test]
    fn test_timestamp_unix_secs() {
        let secs = 1609459200.0; // 2021-01-01 00:00:00 UTC
        let ts = Timestamp::from_unix_secs(secs);
        assert_eq!(ts.as_unix_secs(), secs);
    }

    #[test]
    fn test_timestamp_micros() {
        let micros = 1_609_459_200_000_000u64;
        let ts = Timestamp::from_micros(micros);
        assert_eq!(ts.as_micros(), micros);
    }

    #[test]
    fn test_timestamp_subtract_days() {
        let now = Timestamp::now();
        let yesterday = now.subtract_days(1);
        let diff_secs = now.seconds_since(&yesterday);
        assert!((diff_secs - 86400.0).abs() < 1.0); // ~24 hours
    }

    #[test]
    fn test_timestamp_add_days() {
        let now = Timestamp::now();
        let tomorrow = now.add_days(1);
        let diff_secs = tomorrow.seconds_since(&now);
        assert!((diff_secs - 86400.0).abs() < 1.0); // ~24 hours
    }

    #[test]
    fn test_start_of_day() {
        let ts = Timestamp::from_unix_secs(1609459200.0 + 3600.0); // 2021-01-01 01:00:00
        let start = ts.start_of_day();
        assert_eq!(start.as_unix_secs(), 1609459200.0); // 2021-01-01 00:00:00
    }

    #[test]
    fn test_end_of_day() {
        let ts = Timestamp::from_unix_secs(1609459200.0); // 2021-01-01 00:00:00
        let end = ts.end_of_day();
        let expected = 1609459200.0 + 86400.0 - 0.000001; // 23:59:59.999999
        assert!((end.as_unix_secs() - expected).abs() < 0.001);
    }

    #[test]
    fn test_seconds_since() {
        let t1 = Timestamp::from_unix_secs(1000.0);
        let t2 = Timestamp::from_unix_secs(1010.0);
        assert_eq!(t2.seconds_since(&t1), 10.0);
        assert_eq!(t1.seconds_since(&t2), 0.0); // Past is clamped to 0
    }

    #[test]
    fn test_is_future_past() {
        let now = Timestamp::now();
        let past = now.subtract_days(1);
        let future = now.add_days(1);

        assert!(past.is_past());
        assert!(!past.is_future());
        assert!(future.is_future());
        assert!(!future.is_past());
    }

    #[test]
    fn test_timestamp_bytes() {
        let ts = Timestamp::now();
        let bytes = ts.to_bytes();
        let restored = Timestamp::from_bytes(&bytes).unwrap();
        assert_eq!(ts, restored);
    }

    #[test]
    fn test_timestamp_invalid_bytes() {
        let result = Timestamp::from_bytes(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_timestamp_ordering() {
        let t1 = Timestamp::from_unix_secs(1000.0);
        let t2 = Timestamp::from_unix_secs(2000.0);
        let t3 = Timestamp::from_unix_secs(1000.0);

        assert!(t1 < t2);
        assert!(t2 > t1);
        assert_eq!(t1, t3);
    }
}
