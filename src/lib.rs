//! Ananke — GPU/CPU-aware model proxy daemon.

pub mod activity;
pub mod allocator;
pub mod app_state;
pub mod config;
pub mod daemon;
pub mod db;
pub mod devices;
pub mod errors;
pub mod openai_api;
pub mod proxy;
pub mod retention;
pub mod service_registry;
pub mod signals;
pub mod snapshotter;
pub mod state;
pub mod supervise;
