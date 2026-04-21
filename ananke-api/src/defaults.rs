//! Default ports and endpoints shared between the `ananke` daemon and the
//! `anankectl` CLI. Keeping them in the shared crate ensures a change to the
//! daemon's default listen addresses shows up in the CLI's default endpoint
//! at the same time.
//!
//! These values also mirror the deployment defaults used by the NixOS
//! module that packages `ananke`, so a zero-configuration binary and a
//! zero-configuration systemd unit agree out of the box.

/// Default listen address for the OpenAI-compatible API
/// (`/v1/chat/completions`, `/v1/models`, etc.).
pub const OPENAI_LISTEN: &str = "127.0.0.1:7070";

/// Default listen address for the management API (`/api/services`,
/// `/api/config`, `/api/devices`, etc.).
pub const MANAGEMENT_LISTEN: &str = "127.0.0.1:7071";

/// Default base URL for `anankectl` to reach the management API on a
/// loopback-local daemon.
pub const MANAGEMENT_ENDPOINT: &str = "http://127.0.0.1:7071";
