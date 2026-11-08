//! Embed the built dashboard (`frontend/dist/**`) into the daemon binary
//! and serve it from the management router's fallback handler.
//!
//! Requests for `/api/*`, `/v1/*`, etc. hit their normal axum routes
//! first; anything else goes through [`serve_asset`], which looks up the
//! requested path in the embedded bundle and falls back to `index.html`
//! so the frontend's client-side routing (if/when it grows any) still
//! resolves.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use axum::{
    Router,
    body::Body,
    http::{StatusCode, Uri, header},
    response::{IntoResponse, Response},
};
use rust_embed::Embed;

#[derive(Embed)]
#[folder = "$CARGO_MANIFEST_DIR/../frontend/dist"]
struct Assets;

pub fn register(router: Router) -> Router {
    router.fallback(serve_asset)
}

/// Identifies the embedded frontend: the package version plus a hash of
/// `index.html`, which names every content-hashed bundle, so it changes
/// exactly when the served UI does. An open tab compares it across
/// reconnects to notice that the daemon was upgraded underneath it.
pub(crate) fn build_id() -> String {
    let version = env!("CARGO_PKG_VERSION");
    match Assets::get("index.html") {
        Some(index) => {
            let hash: String = index.metadata.sha256_hash()[..6]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            format!("{version}+{hash}")
        }
        None => version.to_owned(),
    }
}

async fn serve_asset(uri: Uri) -> Response {
    let raw = uri.path().trim_start_matches('/');
    let path = if raw.is_empty() { "index.html" } else { raw };

    if let Some(file) = Assets::get(path) {
        return asset_response(path, file);
    }
    // SPA fallback: unknown paths return index.html so the browser's
    // address bar stays honest if/when the frontend grows client-side
    // routing. Everything hit here is already-non-API (fallback runs
    // after axum's routing), so we're not shadowing a real endpoint.
    if let Some(file) = Assets::get("index.html") {
        return asset_response("index.html", file);
    }
    StatusCode::NOT_FOUND.into_response()
}

fn asset_response(path: &str, file: rust_embed::EmbeddedFile) -> Response {
    let mime = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime)
        .body(Body::from(file.data.into_owned()))
        // Invariant: the status and single content-type header are fixed
        // valid values, so the response builder cannot fail.
        .unwrap_or_else(|_| unreachable!("response with fixed status and headers builds"))
}
