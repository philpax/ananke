//! `/api/oneshot*` endpoint types — create, list, get, delete.

pub mod create;
pub mod delete;
pub mod get;
pub mod list;

pub use create::{
    OneshotAllocation, OneshotDevices, OneshotHealth, OneshotRequest, OneshotResponse,
};
pub use list::OneshotStatus;
