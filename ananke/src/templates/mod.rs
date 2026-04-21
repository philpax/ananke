//! Template dispatch and rendering.
//!
//! `llama-cpp` and `command` are peer templates; shared logic
//! (placeholder substitution, argv assembly) lives here so future
//! templates don't leak into `supervise::spawn`.

pub mod placeholders;

pub use placeholders::{PlaceholderContext, SubstituteError, substitute, substitute_argv};
