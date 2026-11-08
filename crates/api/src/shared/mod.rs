//! Types shared across multiple endpoints — error envelopes, metadata
//! passthrough, model modality, lifecycle, and service state.

pub mod errors;
pub mod lifecycle;
pub mod metadata;
pub mod modality;
pub mod service_state;

pub use lifecycle::Lifecycle;
pub use metadata::AnankeMetadata;
pub use modality::Modality;
pub use service_state::{DisableReason, ServiceStateKind};
