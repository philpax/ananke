#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::process::ExitCode;

use reqwest::{StatusCode, Url};
use serde::de::DeserializeOwned;

pub struct ApiClient {
    pub endpoint: Url,
    http: reqwest::Client,
}

#[derive(Debug)]
pub enum ApiClientError {
    Connect(reqwest::Error),
    Http {
        status: StatusCode,
        body: String,
    },
    Parse(String),
    Usage(String),
    /// The `--endpoint` URL could not be parsed.
    InvalidEndpoint {
        endpoint: String,
        cause: url::ParseError,
    },
    /// A request path could not be joined onto the endpoint URL.
    InvalidPath {
        endpoint: String,
        path: String,
        cause: url::ParseError,
    },
    /// WebSocket connection failure (e.g. from `--follow`).
    WebSocket(String),
}

impl std::fmt::Display for ApiClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect(e) => write!(f, "connection error: {e}"),
            Self::Http { status, body } => write!(f, "HTTP {status}: {body}"),
            Self::Parse(e) => write!(f, "parse error: {e}"),
            Self::Usage(e) => write!(f, "usage error: {e}"),
            Self::WebSocket(e) => write!(f, "WebSocket error: {e}"),
            Self::InvalidEndpoint { endpoint, cause } => {
                write!(f, "invalid --endpoint URL `{endpoint}`: {cause}")
            }
            Self::InvalidPath {
                endpoint,
                path,
                cause,
            } => write!(f, "invalid API path `{path}` on `{endpoint}`: {cause}"),
        }
    }
}

impl std::error::Error for ApiClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Connect(e) => Some(e),
            Self::InvalidEndpoint { cause, .. } | Self::InvalidPath { cause, .. } => Some(cause),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ApiClientError {
    fn from(e: std::io::Error) -> Self {
        Self::Usage(format!("IO error: {e}"))
    }
}

impl ApiClientError {
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Usage(_) | Self::InvalidEndpoint { .. } | Self::InvalidPath { .. } => {
                ExitCode::from(2)
            }
            Self::Connect(_) | Self::WebSocket(_) => ExitCode::from(3),
            _ => ExitCode::from(1),
        }
    }
}

impl ApiClient {
    pub fn new(endpoint: &str) -> Result<Self, ApiClientError> {
        let endpoint = Url::parse(endpoint).map_err(|cause| ApiClientError::InvalidEndpoint {
            endpoint: endpoint.to_string(),
            cause,
        })?;
        Ok(Self {
            endpoint,
            http: reqwest::Client::new(),
        })
    }

    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, ApiClientError> {
        let url = self
            .endpoint
            .join(path)
            .map_err(|cause| ApiClientError::InvalidPath {
                endpoint: self.endpoint.to_string(),
                path: path.to_string(),
                cause,
            })?;
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(ApiClientError::Connect)?;
        self.read_json(resp).await
    }

    pub async fn post_json<T: DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, ApiClientError> {
        let url = self
            .endpoint
            .join(path)
            .map_err(|cause| ApiClientError::InvalidPath {
                endpoint: self.endpoint.to_string(),
                path: path.to_string(),
                cause,
            })?;
        let resp = self
            .http
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(ApiClientError::Connect)?;
        self.read_json(resp).await
    }

    pub async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T, ApiClientError> {
        let url = self
            .endpoint
            .join(path)
            .map_err(|cause| ApiClientError::InvalidPath {
                endpoint: self.endpoint.to_string(),
                path: path.to_string(),
                cause,
            })?;
        let resp = self
            .http
            .post(url)
            .send()
            .await
            .map_err(ApiClientError::Connect)?;
        self.read_json(resp).await
    }

    pub async fn put_body(
        &self,
        path: &str,
        body: String,
        if_match: Option<&str>,
    ) -> Result<(), ApiClientError> {
        let url = self
            .endpoint
            .join(path)
            .map_err(|cause| ApiClientError::InvalidPath {
                endpoint: self.endpoint.to_string(),
                path: path.to_string(),
                cause,
            })?;
        let mut req = self.http.put(url).body(body);
        if let Some(h) = if_match {
            req = req.header(reqwest::header::IF_MATCH, format!("\"{h}\""));
        }
        let resp = req.send().await.map_err(ApiClientError::Connect)?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            Err(ApiClientError::Http { status, body })
        }
    }

    /// The `ws://` (or `wss://`) URL of the daemon's `/api/live` socket.
    pub fn live_url(&self) -> Result<String, ApiClientError> {
        let path = "/api/live";
        let mut url = self
            .endpoint
            .join(path)
            .map_err(|cause| ApiClientError::InvalidPath {
                endpoint: self.endpoint.to_string(),
                path: path.to_string(),
                cause,
            })?;
        let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
        url.set_scheme(scheme).map_err(|()| {
            ApiClientError::Usage(format!("cannot use `{url}` as a WebSocket URL"))
        })?;
        Ok(url.to_string())
    }

    async fn read_json<T: DeserializeOwned>(
        &self,
        resp: reqwest::Response,
    ) -> Result<T, ApiClientError> {
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ApiClientError::Http { status, body });
        }
        resp.json::<T>()
            .await
            .map_err(|e| ApiClientError::Parse(e.to_string()))
    }
}
