//! What kind of model the service exposes through the OpenAI-compatible
//! API. Drives badge rendering in the frontend and lets clients (Discord
//! rotation, RAG indexers) filter the model list by purpose without
//! parsing `metadata.*` strings.
//!
//! Defaults to `Chat` so existing configs and JSON payloads stay
//! byte-identical — the field is elided from the wire when it's `Chat`.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// What kind of model the service exposes through the OpenAI-compatible
/// API. Drives badge rendering in the frontend and lets clients (Discord
/// rotation, RAG indexers) filter the model list by purpose without
/// parsing `metadata.*` strings.
///
/// Defaults to `Chat` so existing configs and JSON payloads stay
/// byte-identical — the field is elided from the wire when it's `Chat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Text generation: `/v1/chat/completions` and `/v1/completions`.
    /// The default for backward compatibility.
    #[default]
    Chat,
    /// Vector embeddings: `/v1/embeddings`. Pooling-only models such as
    /// jina-embeddings-v5, BGE, E5, etc.
    Embedding,
}

impl Modality {
    /// Predicate for `#[serde(skip_serializing_if)]` so the default
    /// (`Chat`) is elided from JSON. Existing chat services then ship
    /// the exact same wire bytes they shipped before this field landed.
    pub fn is_chat(&self) -> bool {
        matches!(self, Modality::Chat)
    }
}
