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
    // Unused today but reserved for commands that validate locally before hitting the API.
    #[expect(dead_code)]
    Usage(String),
}

impl std::fmt::Display for ApiClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect(e) => write!(f, "connection error: {e}"),
            Self::Http { status, body } => write!(f, "HTTP {status}: {body}"),
            Self::Parse(e) => write!(f, "parse error: {e}"),
            Self::Usage(e) => write!(f, "usage error: {e}"),
        }
    }
}

impl std::error::Error for ApiClientError {}

impl ApiClientError {
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Usage(_) => ExitCode::from(2),
            Self::Connect(_) => ExitCode::from(3),
            _ => ExitCode::from(1),
        }
    }
}

impl ApiClient {
    pub fn new(endpoint: &str) -> Self {
        let endpoint = Url::parse(endpoint).expect("valid --endpoint URL");
        Self {
            endpoint,
            http: reqwest::Client::new(),
        }
    }

    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(ApiClientError::Connect)?;
        self.read_json(resp).await
    }

    #[expect(dead_code)]
    pub async fn post_json<T: DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
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
        let url = self.endpoint.join(path).expect("valid path");
        let resp = self
            .http
            .post(url)
            .send()
            .await
            .map_err(ApiClientError::Connect)?;
        self.read_json(resp).await
    }

    #[expect(dead_code)]
    pub async fn delete(&self, path: &str) -> Result<(), ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
        let resp = self
            .http
            .delete(url)
            .send()
            .await
            .map_err(ApiClientError::Connect)?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            Err(ApiClientError::Http { status, body })
        }
    }

    #[expect(dead_code)]
    pub async fn put_body(
        &self,
        path: &str,
        body: String,
        if_match: Option<&str>,
    ) -> Result<(), ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
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
