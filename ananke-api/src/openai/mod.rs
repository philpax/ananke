//! OpenAI-compatible API wire types.
//!
//! These types live in the shared wire-types crate so both the daemon
//! and `anankectl` can use them. The daemon's `api/openai/schema.rs`
//! re-exports from here.

pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod models;
pub mod unimplemented;

pub use chat::ChatCompletionEnvelope;
pub use completions::CompletionEnvelope;
pub use embeddings::EmbeddingEnvelope;
pub use models::{ModelListing, ModelsResponse};
use utoipa::openapi::schema::{AdditionalProperties, ObjectBuilder};

/// Schema for the `extra` passthrough field on OpenAI request envelopes:
/// marks the object as accepting any additional properties, reflecting
/// that all non-`model` keys are forwarded verbatim to the upstream service.
pub fn passthrough_schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
    ObjectBuilder::new()
        .additional_properties(Some(AdditionalProperties::FreeForm(true)))
        .into()
}
