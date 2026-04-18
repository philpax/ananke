//! Shared helpers for integration tests.

// Not every integration test binary uses every symbol from this module.
#![allow(dead_code)]

pub mod echo_server;

use std::net::TcpListener;

/// Binds an ephemeral port and returns it, releasing the listener before returning.
///
/// There is a small TOCTOU window between releasing the listener and the test
/// code binding the same port; in practice this is harmless in CI because the
/// port is chosen by the OS from the ephemeral range and is not reused immediately.
pub fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local_addr").port()
}
