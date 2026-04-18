//! Multi-shard GGUF helpers (implemented in Task 3).

use std::path::Path;

use super::reader::ReadError;
use super::types::GgufSummary;

pub fn read(_path: &Path) -> Result<GgufSummary, ReadError> {
    Err(ReadError("not yet implemented".into()))
}
