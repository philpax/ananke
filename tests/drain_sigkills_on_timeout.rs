use std::{
    sync::{Arc, atomic::AtomicU64},
    time::Duration,
};

use ananke::drain::{DrainConfig, DrainReason, drain_pipeline};

#[tokio::test(flavor = "current_thread")]
async fn sigkill_after_sigterm_grace() {
    // Child traps SIGTERM so it never exits gracefully; drain must SIGKILL.
    let mut child = tokio::process::Command::new("/bin/sh")
        .args(["-c", "trap '' TERM; sleep 60"])
        .spawn()
        .unwrap();

    let counter = Arc::new(AtomicU64::new(0));
    let cfg = DrainConfig {
        max_request_duration: Duration::from_millis(100),
        drain_timeout: Duration::from_millis(50),
        extended_stream_drain: Duration::from_millis(50),
        // Short grace so the test completes quickly.
        sigterm_grace: Duration::from_millis(500),
    };
    drain_pipeline(&mut child, &cfg, counter, DrainReason::Shutdown).await;
    let status = child.wait().await.unwrap();
    // SIGKILL produces a non-success exit status.
    assert!(!status.success(), "child should have been SIGKILL'd");
}
