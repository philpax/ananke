//! End-to-end test: proxy forwards a plain request to the echo server.
#![cfg(feature = "test-fakes")]

mod common;

use std::net::SocketAddr;

use tokio::sync::watch;

#[tokio::test]
async fn proxy_forwards_plain_request() {
    // Bind echo server on an ephemeral port.
    let echo_port = common::free_port();
    let echo_addr: SocketAddr = format!("127.0.0.1:{echo_port}").parse().unwrap();
    let (echo_shutdown_tx, echo_shutdown_rx) = watch::channel(false);
    let _echo_state = common::echo_server::EchoState::default();
    tokio::spawn(common::echo_server::serve(
        echo_addr,
        _echo_state,
        echo_shutdown_rx,
    ));
    let _echo_shutdown = echo_shutdown_tx;

    // Bind proxy on another ephemeral port.
    let proxy_port = common::free_port();
    let proxy_addr: SocketAddr = format!("127.0.0.1:{proxy_port}").parse().unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    tokio::spawn(ananke::api::proxy::serve(
        proxy_addr,
        echo_port,
        shutdown_rx,
    ));

    // Give both servers a moment to start accepting.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://127.0.0.1:{proxy_port}/anything");
    let resp = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .expect("request to proxy");

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("body");
    assert_eq!(body, "hello");

    // Shut down the proxy cleanly.
    let _ = shutdown_tx.send(true);
}
