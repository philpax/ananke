//! Service supervision: per-service tokio tasks, child lifetimes, health loops.

pub mod health;
pub mod logs;
pub mod orphans;
pub mod spawn;

pub use spawn::{SpawnConfig, render_argv};
