//! `GET /api/config` response body.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `GET /api/config` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ConfigResponse {
    /// Raw TOML content.
    pub content: String,
    /// SHA-256 of `content`, base64-encoded; used as an `If-Match` value on PUT.
    pub hash: String,
    /// Whether the config file can be written to. `false` when the file
    /// is on a read-only store (e.g. NixOS-managed); the frontend should
    /// start the editor in read-only mode.
    pub writable: bool,
}
