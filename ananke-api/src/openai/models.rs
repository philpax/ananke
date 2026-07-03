//! `GET /v1/models` — list available models.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::shared::{metadata::AnankeMetadata, modality::Modality};

/// One entry in the `GET /v1/models` response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListing {
    /// Model id (the service name).
    pub id: String,
    /// Always `"model"`.
    pub object: &'static str,
    /// Creation timestamp (always 0 for ananke-managed models).
    pub created: u64,
    /// Always `"ananke"`.
    pub owned_by: &'static str,
    /// What kind of OpenAI endpoint this model serves. Non-standard
    /// OpenAI field, elided when [`Modality::Chat`] (the default), so
    /// strict OpenAI clients see exactly what they saw before this
    /// field landed; embedding clients can filter on it.
    #[serde(default, skip_serializing_if = "Modality::is_chat")]
    pub modality: Modality,
    /// Passthrough entries from `[[service]] metadata.*`. Non-standard
    /// OpenAI field, elided when empty; strict OpenAI clients ignore it.
    #[serde(default, skip_serializing_if = "AnankeMetadata::is_empty")]
    #[schema(value_type = Object)]
    pub ananke_metadata: AnankeMetadata,
}

/// `GET /v1/models` response body.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    /// Always `"list"`.
    pub object: &'static str,
    /// Available models.
    pub data: Vec<ModelListing>,
}
