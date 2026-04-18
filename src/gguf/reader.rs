//! Single-file GGUF reader (implemented in Task 2).

use std::path::Path;

use super::types::GgufSummary;

pub struct ReadError(pub String);

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gguf read failed: {}", self.0)
    }
}

impl std::fmt::Debug for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReadError({})", self.0)
    }
}

impl std::error::Error for ReadError {}

pub fn read_single(_path: &Path) -> Result<GgufSummary, ReadError> {
    Err(ReadError("not yet implemented".into()))
}
