//! Request and response envelopes for the unified OpenAI listener.
//!
//! The daemon does not fully interpret OpenAI chat/completion/embedding
//! bodies; it extracts `model` and forwards the rest. Envelopes surface
//! only `model` in the OpenAPI schema, plus `#[serde(flatten)]` to
//! capture arbitrary other keys.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListing {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
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
