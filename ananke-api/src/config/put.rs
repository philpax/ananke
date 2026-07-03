//! `PUT /api/config` — update the daemon config.
//!
//! Returns `202 Accepted` on success (no body). Error codes are surfaced
//! via the shared [`ApiErrorBody`](crate::shared::errors::ApiErrorBody).

// The PUT endpoint has no bespoke request or response body type: the
// request body is raw TOML (`String`), and the success response is a
// bare `202` with no body. This module exists so the per-endpoint
// module structure is uniform and there's a home for any future
// bespoke types.
