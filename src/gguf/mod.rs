//! GGUF reader — single-file and sharded.

pub mod reader;
pub mod shards;
pub mod types;

pub use reader::read_single;
pub use shards::read;
pub use types::{GgufSummary, GgufTensor, GgufType, GgufValue};
