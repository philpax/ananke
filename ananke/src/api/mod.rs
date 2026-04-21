//! HTTP API surface: OpenAI-compatible router, management router, OpenAPI
//! schema endpoint, and the per-service reverse proxy.

pub mod frontend;
pub mod management;
pub mod openai;
pub mod openapi;
pub mod proxy;
