The daemon serves its own OpenAPI specification at `GET /api/openapi.json`. This is the same document used to generate this file and the frontend's TypeScript types.

The spec is produced by `utoipa`'s compile-time derive from the `#[utoipa::path]` annotations on the daemon's handler functions and the `#[derive(ToSchema)]` types in the `ananke-api` crate. No daemon state or runtime introspection is involved.
