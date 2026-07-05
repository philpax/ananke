//! Configuration defaults, listen addresses, and documentation descriptors.
//!
//! This crate is a leaf dependency (no external deps beyond serde) so the
//! xtask and CLI can reference config defaults without pulling in the
//! daemon's heavy deps (NVML, rusqlite, frontend build).

#![deny(missing_docs)]

pub mod defaults;
pub mod docs;
