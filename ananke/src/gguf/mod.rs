//! GGUF reader — single-file and sharded.
//!
//! All filesystem interaction flows through a [`crate::system::Fs`] handle,
//! so tests can substitute [`crate::system::InMemoryFs`] preloaded with
//! synthetic bytes. Production calls pass [`crate::system::LocalFs`].

pub mod reader;
pub mod shards;
pub mod types;

pub use reader::{ReadError, read_single};
pub use shards::read;
pub use types::{GgufSummary, GgufTensor, GgufType, GgufValue};
