//! Types shared across multiple endpoints — error envelopes, metadata
//! passthrough, and model modality.

pub mod errors;
pub mod metadata;
pub mod modality;

pub use metadata::AnankeMetadata;
pub use modality::Modality;
