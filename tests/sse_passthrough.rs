//! End-to-end test: proxy streams SSE events without buffering.

mod common;

use std::net::SocketAddr;
use std::time::Instant;

use futures::StreamExt as _;
use tokio::sync::watch;

#[tokio::test]
async fn sse_chunks_arrive_incrementally() {
    let echo_port = common::free_port();
    let echo_addr: SocketAddr = format!("127.0.0.1:{echo_port}").parse().unwrap();
    let _echo_shutdown = common::echo_server::spawn(echo_addr).await;

    let proxy_port = common::free_port();
    let proxy_addr: SocketAddr = format!("127.0.0.1:{proxy_port}").parse().unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    tokio::spawn(ananke::proxy::serve(proxy_addr, echo_port, shutdown_rx));

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://127.0.0.1:{proxy_port}/sse");
    let resp = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .expect("SSE request to proxy");

    assert_eq!(resp.status(), 200);

    let start = Instant::now();
    let mut chunk_count = 0usize;

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        chunk.expect("chunk read");
        chunk_count += 1;
    }

    let elapsed = start.elapsed();

    // The echo server emits 5 chunks 50 ms apart, so total wall time must be
    // at least 150 ms (3 × 50 ms) and we must have received at least 3 chunks.
    assert!(
        elapsed.as_millis() >= 150,
        "expected >=150 ms elapsed, got {elapsed:?}"
    );
    assert!(chunk_count >= 3, "expected >=3 chunks, got {chunk_count}");

    let _ = shutdown_tx.send(true);
}
