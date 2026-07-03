//! 501 Not Implemented stubs for unsupported OpenAI endpoints.
//!
//! The daemon returns `501 Not Implemented` for `/v1/audio/*`,
//! `/v1/images/*`, `/v1/files/*`, `/v1/fine_tuning/*`, and
//! `/v1/batches`. These have no bespoke request/response types — the
//! error is projected through the shared [`ApiErrorBody`].

// This module exists so the per-endpoint module structure is uniform
// and there's a home for any future bespoke types.
