//! HTTP health-probe loop.

use std::time::Duration;

use tokio::sync::watch;
use tracing::debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthOutcome {
    Healthy,
    TimedOut,
    Cancelled,
}

pub struct HealthConfig {
    pub url: String,
    pub probe_interval: Duration,
    pub timeout: Duration,
}

/// Runs until one of: health passes → `Healthy`; total elapsed exceeds
/// `timeout` → `TimedOut`; `cancel` resolves to true → `Cancelled`.
pub async fn wait_healthy(cfg: HealthConfig, mut cancel: watch::Receiver<bool>) -> HealthOutcome {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("reqwest client build");
    let start = tokio::time::Instant::now();

    loop {
        if *cancel.borrow() {
            return HealthOutcome::Cancelled;
        }
        if start.elapsed() >= cfg.timeout {
            return HealthOutcome::TimedOut;
        }

        match client.get(&cfg.url).send().await {
            Ok(resp) if resp.status().is_success() => return HealthOutcome::Healthy,
            Ok(resp) => debug!(status = %resp.status(), url = %cfg.url, "health probe non-2xx"),
            Err(e) => debug!(error = %e, url = %cfg.url, "health probe errored"),
        }

        tokio::select! {
            _ = tokio::time::sleep(cfg.probe_interval) => {}
            _ = cancel.changed() => {
                if *cancel.borrow() { return HealthOutcome::Cancelled; }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        sync::{
            Arc,
            atomic::{AtomicU32, Ordering},
        },
    };

    use bytes::Bytes;
    use http_body_util::Full;
    use hyper::{Request, Response, service::service_fn};
    use hyper_util::{rt::TokioIo, server::conn::auto};
    use tokio::net::TcpListener;

    use super::*;

    async fn spawn_server(status: u16) -> (String, Arc<AtomicU32>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicU32::new(0));
        let count_clone = count.clone();
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let io = TokioIo::new(stream);
                let count = count_clone.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |_req: Request<hyper::body::Incoming>| {
                        let count = count.clone();
                        async move {
                            count.fetch_add(1, Ordering::Relaxed);
                            let resp = Response::builder()
                                .status(status)
                                .body(Full::new(Bytes::from("ok")))
                                .unwrap();
                            Ok::<_, Infallible>(resp)
                        }
                    });
                    let _ = auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                        .serve_connection(io, svc)
                        .await;
                });
            }
        });
        (format!("http://{addr}/health"), count)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn returns_healthy_on_2xx() {
        let (url, _) = spawn_server(200).await;
        let (tx, rx) = watch::channel(false);
        let outcome = wait_healthy(
            HealthConfig {
                url,
                probe_interval: Duration::from_millis(50),
                timeout: Duration::from_secs(5),
            },
            rx,
        )
        .await;
        drop(tx);
        assert_eq!(outcome, HealthOutcome::Healthy);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn times_out_on_always_500() {
        let (url, _) = spawn_server(500).await;
        let (_, rx) = watch::channel(false);
        let outcome = wait_healthy(
            HealthConfig {
                url,
                probe_interval: Duration::from_millis(50),
                timeout: Duration::from_millis(300),
            },
            rx,
        )
        .await;
        assert_eq!(outcome, HealthOutcome::TimedOut);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_when_signalled() {
        let (url, _) = spawn_server(500).await;
        let (tx, rx) = watch::channel(false);
        let task = tokio::spawn(wait_healthy(
            HealthConfig {
                url,
                probe_interval: Duration::from_millis(500),
                timeout: Duration::from_secs(10),
            },
            rx,
        ));
        tokio::time::sleep(Duration::from_millis(100)).await;
        tx.send(true).unwrap();
        let outcome = task.await.unwrap();
        assert_eq!(outcome, HealthOutcome::Cancelled);
    }
}
