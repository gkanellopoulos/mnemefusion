//! File format definitions
//!
//! Defines the .mfdb file format structure and validation.

use crate::{Error, Result};
use std::time::{SystemTime, UNIX_EPOCH};

/// Magic number for MnemeFusion database files
pub const MAGIC: &[u8; 4] = b"MFDB";

/// Current file format version
pub const VERSION: u32 = 1;

/// File header for .mfdb files
///
/// The header is 64 bytes and contains file metadata and version information.
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic number: "MFDB"
    pub magic: [u8; 4],

    /// File format version
    pub version: u32,

    /// Flags for future use
    pub flags: u64,

    /// Creation timestamp (Unix seconds)
    pub created_at: u64,

    /// Last modified timestamp (Unix seconds)
    pub modified_at: u64,

    /// Reserved space for future use
    pub reserved: [u8; 32],
}

impl FileHeader {
    /// Create a new file header with current timestamp
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time before UNIX_EPOCH")
            .as_secs();

        Self {
            magic: *MAGIC,
            version: VERSION,
            flags: 0,
            created_at: now,
            modified_at: now,
            reserved: [0; 32],
        }
    }

    /// Validate the file header
    ///
    /// Checks magic number, version compatibility, and basic sanity checks.
    pub fn validate(&self) -> Result<()> {
        // Check magic number
        if &self.magic != MAGIC {
            return Err(Error::InvalidFormat("Invalid magic number - file may be corrupted or not a MnemeFusion database"));
        }

        // Check version
        if self.version == 0 {
            return Err(Error::DatabaseCorruption("Invalid version 0".to_string()));
        }

        if self.version > VERSION {
            return Err(Error::UnsupportedVersion(self.version, VERSION));
        }

        // Check timestamps are reasonable (after 2020-01-01 and before 2100-01-01)
        const YEAR_2020: u64 = 1577836800; // 2020-01-01 in Unix seconds
        const YEAR_2100: u64 = 4102444800; // 2100-01-01 in Unix seconds

        if self.created_at < YEAR_2020 || self.created_at > YEAR_2100 {
            return Err(Error::DatabaseCorruption(
                format!("Invalid creation timestamp: {}", self.created_at)
            ));
        }

        if self.modified_at < YEAR_2020 || self.modified_at > YEAR_2100 {
            return Err(Error::DatabaseCorruption(
                format!("Invalid modification timestamp: {}", self.modified_at)
            ));
        }

        // modified_at should be >= created_at
        if self.modified_at < self.created_at {
            return Err(Error::DatabaseCorruption(
                format!("Modified timestamp ({}) before created timestamp ({})",
                    self.modified_at, self.created_at)
            ));
        }

        Ok(())
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];

        // Magic (4 bytes)
        bytes[0..4].copy_from_slice(&self.magic);

        // Version (4 bytes)
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());

        // Flags (8 bytes)
        bytes[8..16].copy_from_slice(&self.flags.to_le_bytes());

        // Created at (8 bytes)
        bytes[16..24].copy_from_slice(&self.created_at.to_le_bytes());

        // Modified at (8 bytes)
        bytes[24..32].copy_from_slice(&self.modified_at.to_le_bytes());

        // Reserved (32 bytes)
        bytes[32..64].copy_from_slice(&self.reserved);

        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 64 {
            return Err(Error::InvalidFormat("Header too short"));
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);

        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let flags = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let created_at = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19],
            bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let modified_at = u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]);

        let mut reserved = [0u8; 32];
        reserved.copy_from_slice(&bytes[32..64]);

        let header = Self {
            magic,
            version,
            flags,
            created_at,
            modified_at,
            reserved,
        };

        header.validate()?;
        Ok(header)
    }

    /// Update the modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time before UNIX_EPOCH")
            .as_secs();
    }
}

impl Default for FileHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_header_new() {
        let header = FileHeader::new();
        assert_eq!(&header.magic, MAGIC);
        assert_eq!(header.version, VERSION);
        assert_eq!(header.flags, 0);
        assert!(header.created_at > 0);
        assert_eq!(header.created_at, header.modified_at);
    }

    #[test]
    fn test_file_header_validate() {
        let header = FileHeader::new();
        assert!(header.validate().is_ok());

        let mut bad_header = header.clone();
        bad_header.magic = *b"XXXX";
        assert!(bad_header.validate().is_err());

        let mut bad_header = header.clone();
        bad_header.version = VERSION + 1;
        assert!(bad_header.validate().is_err());
    }

    #[test]
    fn test_file_header_serialization() {
        let header = FileHeader::new();
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 64);

        let restored = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(restored.magic, header.magic);
        assert_eq!(restored.version, header.version);
        assert_eq!(restored.flags, header.flags);
        assert_eq!(restored.created_at, header.created_at);
        assert_eq!(restored.modified_at, header.modified_at);
    }

    #[test]
    fn test_file_header_invalid_bytes() {
        let short_bytes = [0u8; 32];
        assert!(FileHeader::from_bytes(&short_bytes).is_err());

        let mut bad_magic = [0u8; 64];
        bad_magic[0..4].copy_from_slice(b"XXXX");
        assert!(FileHeader::from_bytes(&bad_magic).is_err());
    }

    #[test]
    fn test_file_header_touch() {
        let mut header = FileHeader::new();
        let original_modified = header.modified_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        header.touch();

        assert!(header.modified_at >= original_modified);
    }

    #[test]
    fn test_magic_constant() {
        assert_eq!(MAGIC, b"MFDB");
    }

    #[test]
    fn test_version_constant() {
        assert_eq!(VERSION, 1);
    }

    #[test]
    fn test_header_validation_corrupted_version() {
        let mut header = FileHeader::new();
        header.version = 0;
        let err = header.validate().unwrap_err();
        assert!(matches!(err, Error::DatabaseCorruption(_)));
    }

    #[test]
    fn test_header_validation_timestamp_out_of_range() {
        // Test created_at too old
        let mut header = FileHeader::new();
        header.created_at = 100; // Too old (before 2020)
        let err = header.validate().unwrap_err();
        assert!(matches!(err, Error::DatabaseCorruption(_)));

        // Test created_at too far in future
        let mut header = FileHeader::new();
        header.created_at = 5000000000; // After 2100
        let err = header.validate().unwrap_err();
        assert!(matches!(err, Error::DatabaseCorruption(_)));

        // Test modified_at too old
        let mut header = FileHeader::new();
        header.modified_at = 100;
        let err = header.validate().unwrap_err();
        assert!(matches!(err, Error::DatabaseCorruption(_)));
    }

    #[test]
    fn test_header_validation_modified_before_created() {
        let mut header = FileHeader::new();
        header.created_at = 1700000000;
        header.modified_at = 1600000000; // Before created_at
        let err = header.validate().unwrap_err();
        assert!(matches!(err, Error::DatabaseCorruption(_)));
    }

    #[test]
    fn test_header_validation_unsupported_future_version() {
        let mut header = FileHeader::new();
        header.version = VERSION + 10;
        let err = header.validate().unwrap_err();
        assert!(matches!(err, Error::UnsupportedVersion(_, _)));
    }

    #[test]
    fn test_header_from_bytes_truncated() {
        // Test with truncated bytes
        let short_bytes = [0u8; 32]; // Less than 64 bytes
        let err = FileHeader::from_bytes(&short_bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn test_header_from_bytes_bad_magic() {
        let mut bytes = FileHeader::new().to_bytes();
        bytes[0..4].copy_from_slice(b"XXXX"); // Corrupt magic number
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }
}
