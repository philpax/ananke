//! `GET /metrics` — Prometheus text-format endpoint.
//!
//! This endpoint serves Prometheus text format (not JSON), so it has no
//! bespoke schema type. The handler is registered in the daemon; the only
//! thing this module documents is that the endpoint exists and returns
//! `text/plain; version=0.0.4; charset=utf-8`.
