//! Config GET/PUT/validate payloads.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `GET /api/config` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ConfigResponse {
    /// Raw TOML content.
    pub content: String,
    /// SHA-256 of `content`, base64-encoded; used as an `If-Match` value on PUT.
    pub hash: String,
}

/// `POST /api/config/validate` request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ConfigValidateRequest {
    /// Raw TOML to validate without persisting.
    pub content: String,
}

/// `POST /api/config/validate` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ConfigValidateResponse {
    /// `true` iff no errors were found.
    pub valid: bool,
    /// Structured validation errors (span-annotated where possible).
    pub errors: Vec<ValidationError>,
}

/// One validation error from the config parser.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ValidationError {
    /// 1-based line number in the TOML source.
    pub line: u32,
    /// 1-based column number in the TOML source.
    pub column: u32,
    /// Human-readable message.
    pub message: String,
}
