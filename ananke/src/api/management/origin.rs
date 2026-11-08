//! Same-origin check for the management WebSockets.
//!
//! Browsers let any page open a WebSocket to any host, and the management
//! listener has no authentication, so without this a page on another site
//! could read the daemon's state through a visitor's browser. Browsers
//! always send `Origin` on a WebSocket handshake; non-browser clients
//! (`anankectl`, scripts, tests) usually do not, and are allowed.

use axum::http::{HeaderMap, header};
use reqwest::Url;

/// Whether the handshake carries no `Origin`, or one whose scheme is http(s)
/// and whose host and port match the `Host` header.
pub(crate) fn same_origin(headers: &HeaderMap) -> bool {
    let Some(origin) = headers.get(header::ORIGIN) else {
        return true;
    };
    let Some(origin) = origin.to_str().ok().and_then(|o| Url::parse(o).ok()) else {
        return false;
    };
    if !matches!(origin.scheme(), "http" | "https") {
        return false;
    }
    let Some(host) = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| Url::parse(&format!("{}://{h}", origin.scheme())).ok())
    else {
        return false;
    };
    host.host() == origin.host() && host.port_or_known_default() == origin.port_or_known_default()
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn headers(origin: Option<&str>, host: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(o) = origin {
            h.insert(header::ORIGIN, HeaderValue::from_str(o).unwrap());
        }
        if let Some(host) = host {
            h.insert(header::HOST, HeaderValue::from_str(host).unwrap());
        }
        h
    }

    #[test]
    fn a_missing_origin_is_allowed() {
        assert!(same_origin(&headers(None, Some("127.0.0.1:7071"))));
        assert!(same_origin(&headers(None, None)));
    }

    #[test]
    fn a_matching_origin_is_allowed() {
        assert!(same_origin(&headers(
            Some("http://127.0.0.1:7071"),
            Some("127.0.0.1:7071")
        )));
        assert!(same_origin(&headers(
            Some("https://ananke.example"),
            Some("ananke.example")
        )));
        // An explicit default port matches an implied one.
        assert!(same_origin(&headers(
            Some("http://ananke.example"),
            Some("ananke.example:80")
        )));
    }

    #[test]
    fn a_mismatched_origin_is_rejected() {
        for (origin, host) in [
            ("http://evil.example", "127.0.0.1:7071"),
            ("http://127.0.0.1:5173", "127.0.0.1:7071"),
            ("null", "127.0.0.1:7071"),
            ("file:///index.html", "127.0.0.1:7071"),
        ] {
            assert!(
                !same_origin(&headers(Some(origin), Some(host))),
                "{origin} vs {host}"
            );
        }
    }

    #[test]
    fn an_origin_without_a_host_is_rejected() {
        assert!(!same_origin(&headers(Some("http://127.0.0.1:7071"), None)));
    }
}
