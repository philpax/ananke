//! `/api/config*` endpoint types — get, put, validate.

pub mod get;
pub mod put;
pub mod validate;

pub use get::ConfigResponse;
pub use validate::{ConfigValidateRequest, ConfigValidateResponse, ValidationError};
