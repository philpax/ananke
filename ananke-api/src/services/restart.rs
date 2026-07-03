//! `POST /api/services/{name}/restart` response body.
//!
//! Restart reuses [`StartResponse`](crate::services::start::StartResponse)
//! because the operation is a stop-then-start; the response shape is
//! identical to `POST /api/services/{name}/start`.

/// Alias of [`crate::services::start::StartResponse`] used by the restart
/// endpoint.
pub type RestartResponse = crate::services::start::StartResponse;
