//! Request and response envelopes for the unified OpenAI listener.
//!
//! The daemon does not fully interpret OpenAI chat/completion/embedding
//! bodies; it extracts `model` and forwards the rest. Envelopes surface
//! only `model` in the OpenAPI schema, plus `#[serde(flatten)]` to
//! capture arbitrary other keys.

use ananke_api::AnankeMetadata;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListing {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    /// Passthrough entries from `[[service]] metadata.*`. Non-standard
    /// OpenAI field, elided when empty; strict OpenAI clients ignore it.
    #[serde(default, skip_serializing_if = "AnankeMetadata::is_empty")]
    #[schema(value_type = Object)]
    pub ananke_metadata: AnankeMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelListing>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionEnvelope {
    pub model: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CompletionEnvelope {
    pub model: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingEnvelope {
    pub model: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}
