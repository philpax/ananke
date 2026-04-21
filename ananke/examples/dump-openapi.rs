//! Print the aggregated OpenAPI JSON document to stdout.
//!
//! Used by `frontend/` to regenerate the TypeScript types from the
//! current Rust handler + schema annotations:
//!
//! ```sh
//! cargo run --example dump-openapi > /tmp/openapi.json
//! cd frontend && npm run gen-types
//! ```
//!
//! The document is produced by `utoipa`'s compile-time derive; no
//! daemon state, config, or listener is involved.

use ananke::api::openapi::AnankeApi;
use utoipa::OpenApi;

fn main() {
    let doc = AnankeApi::openapi();
    let rendered = serde_json::to_string_pretty(&doc).expect("serialize OpenAPI doc");
    println!("{rendered}");
}
