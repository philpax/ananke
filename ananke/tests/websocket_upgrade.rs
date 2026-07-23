//! End-to-end test: the proxy forwards 101 Switching Protocols handshakes
//! with the upstream's `Connection: Upgrade` header verbatim and then
//! splices bytes between the client and upstream sockets.
//!
//! The regression this guards against: aiohttp's WebSocket client validates
//! the response `Connection` header with exact equality (after lowercasing)
//! against `"upgrade"`. A proxy that rewrites it to `keep-alive`,
//! appends `keep-alive` to the list, or strips it entirely (letting hyper's
//! HTTP/1 encoder fill in its default) breaks ComfyUI and every other
//! aiohttp-served WebSocket endpoint behind the proxy.
#![cfg(feature = "test-fakes")]

mod common;

use std::{
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use ananke::api::proxy;
use futures::future;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::watch,
};

/// Tiny hand-rolled HTTP/1 server that answers exactly one WebSocket
/// handshake and then echoes whatever bytes the client sends. The
/// `Connection: Upgrade` line is emitted on its own — no extra tokens — to
/// match the value ComfyUI's aiohttp server sends.
async fn run_ws_backend(listener: TcpListener) {
    let (mut sock, _) = listener.accept().await.expect("accept");
    let mut buf = vec![0u8; 4096];
    let mut total = 0;
    loop {
        let n = sock.read(&mut buf[total..]).await.expect("read req");
        if n == 0 {
            return;
        }
        total += n;
        if buf[..total].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let response = b"HTTP/1.1 101 Switching Protocols\r\n\
                     Upgrade: websocket\r\n\
                     Connection: Upgrade\r\n\
                     Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\
                     \r\n";
    sock.write_all(response).await.expect("write 101");

    let mut frame = [0u8; 4];
    if sock.read_exact(&mut frame).await.is_ok() {
        let _ = sock.write_all(&frame).await;
    }
}

#[tokio::test]
async fn websocket_handshake_preserves_connection_header() {
    let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend_port = backend.local_addr().unwrap().port();
    let backend_task = tokio::spawn(run_ws_backend(backend));

    let proxy_port = common::free_port();
    let proxy_addr: SocketAddr = format!("127.0.0.1:{proxy_port}").parse().unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    tokio::spawn(proxy::serve(proxy_addr, backend_port, shutdown_rx));

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    let handshake = b"GET /ws HTTP/1.1\r\n\
                      Host: 127.0.0.1\r\n\
                      Connection: Upgrade\r\n\
                      Upgrade: websocket\r\n\
                      Sec-WebSocket-Version: 13\r\n\
                      Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
                      \r\n";
    stream.write_all(handshake).await.unwrap();

    let mut buf = vec![0u8; 4096];
    let mut total = 0;
    let header_end = loop {
        let n = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf[total..]))
            .await
            .expect("read timeout")
            .expect("read response");
        assert!(n > 0, "EOF before headers");
        total += n;
        if let Some(idx) = buf[..total].windows(4).position(|w| w == b"\r\n\r\n") {
            break idx + 4;
        }
    };

    let header_text = std::str::from_utf8(&buf[..header_end]).expect("utf8 headers");
    let status_line = header_text.lines().next().unwrap();
    assert!(
        status_line.contains(" 101 "),
        "expected 101 status, got: {status_line}"
    );

    let connection_value = header_text
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("connection") {
                Some(value.trim().to_string())
            } else {
                None
            }
        })
        .expect("Connection header in response");

    // aiohttp's check is `headers.get("connection", "").lower() == "upgrade"`.
    // Any extra token (e.g. "keep-alive, Upgrade") breaks the handshake even
    // though it is RFC-6455-compliant.
    assert_eq!(
        connection_value.to_ascii_lowercase(),
        "upgrade",
        "Connection header must be exactly `Upgrade`; got `{connection_value}`"
    );

    let upgrade_value = header_text
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("upgrade") {
                Some(value.trim().to_string())
            } else {
                None
            }
        })
        .expect("Upgrade header in response");
    assert_eq!(upgrade_value.to_ascii_lowercase(), "websocket");

    // Splice check: write four bytes after the handshake, expect them back.
    let payload = b"ping";
    stream.write_all(payload).await.unwrap();
    let mut echo = [0u8; 4];
    tokio::time::timeout(Duration::from_secs(2), stream.read_exact(&mut echo))
        .await
        .expect("echo timeout")
        .expect("echo read");
    assert_eq!(&echo, payload);

    let _ = shutdown_tx.send(true);
    let _ = backend_task.await;
}

/// Backend that holds a WebSocket open until the client either sends
/// `"close"` or half-closes its write end. Used to assert that the
/// proxy keeps its lifecycle bookkeeping live for the full session.
async fn run_holding_ws_backend(listener: TcpListener) {
    let (mut sock, _) = listener.accept().await.expect("accept");
    let mut buf = vec![0u8; 4096];
    let mut total = 0;
    loop {
        let n = sock.read(&mut buf[total..]).await.expect("read req");
        if n == 0 {
            return;
        }
        total += n;
        if buf[..total].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let response = b"HTTP/1.1 101 Switching Protocols\r\n\
                     Upgrade: websocket\r\n\
                     Connection: Upgrade\r\n\
                     Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\
                     \r\n";
    sock.write_all(response).await.expect("write 101");

    let mut chunk = vec![0u8; 64];
    loop {
        match sock.read(&mut chunk).await {
            Ok(0) => return,
            Ok(n) => {
                if chunk[..n].windows(5).any(|w| w == b"close") {
                    return;
                }
                let _ = sock.write_all(&chunk[..n]).await;
            }
            Err(_) => return,
        }
    }
}

#[tokio::test]
async fn websocket_session_holds_inflight_and_pings_activity() {
    let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend_port = backend.local_addr().unwrap().port();
    let backend_task = tokio::spawn(run_holding_ws_backend(backend));

    let proxy_port = common::free_port();
    let proxy_addr: SocketAddr = format!("127.0.0.1:{proxy_port}").parse().unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Lifecycle observability:
    // * `inflight_counter` mirrors the per-service drain counter.
    // * `before_request_calls` increments once per HTTP request — proves
    //   the upgrade handshake is treated as a request, not a magical
    //   side channel that bypasses the activity hook.
    // * `activity_pings` is bumped by the heartbeat the proxy spawns
    //   for an active WebSocket session.
    let inflight_counter = Arc::new(AtomicU64::new(0));
    let before_request_calls = Arc::new(AtomicUsize::new(0));
    let activity_pings = Arc::new(AtomicUsize::new(0));

    let before_request: Arc<
        dyn Fn() -> futures::future::BoxFuture<'static, Option<proxy::ProxyError>> + Send + Sync,
    > = {
        let calls = before_request_calls.clone();
        Arc::new(move || {
            calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(future::ready(None))
        })
    };
    let activity_ping: Arc<dyn Fn() + Send + Sync> = {
        let pings = activity_pings.clone();
        Arc::new(move || {
            pings.fetch_add(1, Ordering::Relaxed);
        })
    };

    tokio::spawn(proxy::serve_with_activity(
        proxy_addr,
        backend_port,
        shutdown_rx,
        before_request,
        inflight_counter.clone(),
        activity_ping,
        None,
    ));

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    let handshake = b"GET /ws HTTP/1.1\r\n\
                      Host: 127.0.0.1\r\n\
                      Connection: Upgrade\r\n\
                      Upgrade: websocket\r\n\
                      Sec-WebSocket-Version: 13\r\n\
                      Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
                      \r\n";
    stream.write_all(handshake).await.unwrap();

    let mut buf = vec![0u8; 4096];
    let mut total = 0;
    loop {
        let n = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf[total..]))
            .await
            .expect("read timeout")
            .expect("read response");
        assert!(n > 0, "EOF before headers");
        total += n;
        if buf[..total].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }

    assert_eq!(
        before_request_calls.load(Ordering::Relaxed),
        1,
        "the WebSocket handshake must run through before_request like any other request"
    );

    // Drive a round-trip through the splice so we know both halves are
    // connected and the session is genuinely live (not just an
    // unfulfilled `try_join!`).
    stream.write_all(b"hello").await.unwrap();
    let mut echo = [0u8; 5];
    tokio::time::timeout(Duration::from_secs(2), stream.read_exact(&mut echo))
        .await
        .expect("echo timeout")
        .expect("echo read");
    assert_eq!(&echo, b"hello");

    // While the session is open the long-lived guard must keep the
    // counter at exactly 1. The closure-scoped guard from
    // `serve_with_activity`'s service_fn has long since dropped by now
    // — the only remaining holder is the splice task. A 0 reading
    // here would mean `drain_pipeline` could SIGTERM mid-session.
    assert_eq!(
        inflight_counter.load(Ordering::Relaxed),
        1,
        "WebSocket session must keep one inflight guard alive"
    );

    // Activity heartbeat: the proxy bumps the stamp every
    // `WS_ACTIVITY_PING_INTERVAL` (5 s). Pause real time, advance the
    // virtual clock past three intervals, and verify the ticker fired.
    // Running under default `#[tokio::test]` means the I/O above ran in
    // real time; the local pause-region prevents the heartbeat from
    // also having to wait wall-clock seconds to fire.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(16)).await;
    tokio::task::yield_now().await;
    tokio::time::resume();
    let pings = activity_pings.load(Ordering::Relaxed);
    assert!(
        pings >= 3,
        "expected ≥3 activity pings after 16s, got {pings}"
    );

    // Tell the backend to close the WS. The splice task should observe
    // EOF, copy_bidirectional returns, the session guard drops, and
    // the inflight counter falls back to 0.
    stream.write_all(b"close").await.unwrap();

    // Read until peer closes (drains any residual echo for "close").
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        let mut throwaway = [0u8; 64];
        loop {
            match stream.read(&mut throwaway).await {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
        }
    })
    .await;

    // The splice task drops the guard once `copy_bidirectional` finishes.
    // Poll a few yield+sleep cycles in real time for that to land.
    for _ in 0..50 {
        if inflight_counter.load(Ordering::Relaxed) == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        inflight_counter.load(Ordering::Relaxed),
        0,
        "inflight must return to 0 once the WebSocket closes"
    );

    let _ = shutdown_tx.send(true);
    let _ = backend_task.await;
}
