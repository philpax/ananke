//! Builds the self-contained dashboard into the management listener.
//! Native API and asset misses return 404; dashboard routes use the SPA entry.
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

async fn serve_asset(uri: Uri) -> Response {
    let raw = uri.path().trim_start_matches('/');
    let path = if raw.is_empty() { "index.html" } else { raw };

    if let Some(file) = Assets::get(path) {
        return asset_response(path, file);
    }
    if ["api", "v1", "assets", "fonts"]
        .iter()
        .any(|prefix| raw == *prefix || raw.starts_with(&format!("{prefix}/")))
    {
        return StatusCode::NOT_FOUND.into_response();
    }
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
