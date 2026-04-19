//! Template dispatch and rendering.
//!
//! Phase 1-3 hard-coded `llama-cpp`. Phase 4 introduces `command` as a
//! peer template; shared logic moves here so future templates don't
//! keep leaking into `supervise::spawn`.

pub mod placeholders;

pub use placeholders::{PlaceholderContext, SubstituteError, substitute, substitute_argv};
