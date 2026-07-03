//! Shared HTTP DTOs between the `ananke` daemon and the `anankectl` CLI.
//!
//! The crate is organised into per-endpoint modules (`services::list`,
//! `services::start`, `config::put`, etc.) so each endpoint's request,
//! response, and error types live in one place. Types shared across
//! endpoints live in [`shared`]; types used beyond the HTTP boundary
//! (broadcast bus events, placement verdicts, log rows) live in
//! [`internal`].

#![deny(missing_docs)]

pub mod config;
pub mod devices;
pub mod events;
pub mod info;
pub mod internal;
pub mod metrics;
pub mod oneshot;
pub mod openai;
pub mod services;
pub mod shared;
