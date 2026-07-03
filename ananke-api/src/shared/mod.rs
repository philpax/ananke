//! Types shared across multiple endpoints — error envelopes, metadata
//! passthrough, model modality, and default listen addresses.

pub mod defaults;
pub mod errors;
pub mod metadata;
pub mod modality;

pub use metadata::AnankeMetadata;
pub use modality::Modality;
