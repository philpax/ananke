//! Ananke — GPU/CPU-aware model proxy daemon.

pub mod allocator;
pub mod api;
pub mod config;
pub mod daemon;
pub mod db;
pub mod devices;
pub mod errors;
pub mod estimator;
pub mod gguf;
pub mod oneshot;
pub mod supervise;
pub mod system;
pub mod templates;
pub mod tracking;
