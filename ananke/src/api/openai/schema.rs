//! Request and response envelopes for the unified OpenAI listener.
//!
//! The schema types now live in `ananke-api` (`ananke_api::openai`) so
//! they're part of the shared wire-types crate alongside all other
//! endpoint types. This module re-exports them for the daemon's existing
//! `use crate::api::openai::schema::*` import paths.

pub use ananke_api::openai::{
    ChatCompletionEnvelope, CompletionEnvelope, EmbeddingEnvelope, ModelListing, ModelsResponse,
};
