//! A client for `/api/live` in integration tests. Every wait is a frame
//! arriving under a deadline, never a sleep.
#![cfg(feature = "test-fakes")]
// Not every integration test binary uses every symbol in this module.
#![allow(dead_code)]

use std::time::Duration;

use ananke_api::live::{LiveMessage, LiveRequest};
use futures::{SinkExt, StreamExt};
use tokio::{net::TcpStream, time::timeout};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};

use crate::common::TestHarness;

pub type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

pub const DEADLINE: Duration = Duration::from_secs(5);

/// Connect and consume the hello, which is always the first frame.
pub async fn connect(h: &TestHarness) -> Socket {
    let addr = h.spawn_management_server().await;
    let (mut socket, _) = connect_async(format!("ws://{addr}/api/live"))
        .await
        .unwrap();
    match next_message(&mut socket, DEADLINE).await {
        Some(LiveMessage::Hello { build }) => {
            assert!(build.starts_with(env!("CARGO_PKG_VERSION")), "{build}");
        }
        other => panic!("expected a hello first, got {other:?}"),
    }
    socket
}

pub async fn send(socket: &mut Socket, request: LiveRequest) {
    let text = serde_json::to_string(&request).unwrap();
    socket.send(Message::Text(text.into())).await.unwrap();
}

/// The next server message, skipping control frames.
pub async fn next_message(socket: &mut Socket, within: Duration) -> Option<LiveMessage> {
    timeout(within, async {
        loop {
            match socket.next().await {
                Some(Ok(Message::Text(text))) => {
                    return serde_json::from_str::<LiveMessage>(&text).unwrap();
                }
                Some(Ok(_)) => {}
                other => panic!("the socket ended: {other:?}"),
            }
        }
    })
    .await
    .ok()
}
