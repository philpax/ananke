//! `POST /v1/embeddings` — request envelope.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::openai::passthrough_schema;

/// `POST /v1/embeddings` request envelope.
///
/// The daemon only interprets the `model` field; all other keys are
/// forwarded verbatim to the upstream service.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingEnvelope {
    /// Model name (maps to an ananke service name).
    pub model: String,
    /// Arbitrary other keys passed through to the upstream.
    #[serde(flatten)]
    #[schema(schema_with = passthrough_schema)]
    pub extra: serde_json::Value,
}
