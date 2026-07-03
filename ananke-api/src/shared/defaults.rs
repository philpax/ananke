//! Default ports and endpoints shared between the `ananke` daemon and the
//! `anankectl` CLI. Keeping them in the shared crate ensures a change to the
//! daemon's default listen addresses shows up in the CLI's default endpoint
//! at the same time, and gives packaging (systemd units, Docker images,
//! etc.) one place to read from when generating default manifests.

/// Default listen address for the OpenAI-compatible API
/// (`/v1/chat/completions`, `/v1/models`, etc.).
pub const OPENAI_LISTEN: &str = "127.0.0.1:7070";

/// Default listen address for the management API (`/api/services`,
/// `/api/config`, `/api/devices`, etc.).
pub const MANAGEMENT_LISTEN: &str = "127.0.0.1:7071";

/// Default base URL for `anankectl` to reach the management API on a
/// loopback-local daemon.
pub const MANAGEMENT_ENDPOINT: &str = "http://127.0.0.1:7071";
