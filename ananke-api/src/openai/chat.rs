//! `POST /v1/chat/completions` — request envelope.
//!
//! The daemon does not fully interpret OpenAI chat/completion/embedding
//! bodies; it extracts `model` and forwards the rest. This envelope
//! surfaces only `model` in the OpenAPI schema, plus `#[serde(flatten)]`
//! to capture arbitrary other keys.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::openai::passthrough_schema;

/// `POST /v1/chat/completions` request envelope.
///
/// The daemon only interprets the `model` field; all other keys are
/// forwarded verbatim to the upstream service.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionEnvelope {
    /// Model name (maps to an ananke service name).
    pub model: String,
    /// Arbitrary other keys passed through to the upstream.
    #[serde(flatten)]
    #[schema(schema_with = passthrough_schema)]
    pub extra: serde_json::Value,
}
