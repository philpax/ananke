use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use ananke::drain::{DrainConfig, DrainReason, drain_pipeline};

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn waits_for_inflight_zero() {
    // Fake child: a shell sleep that the drain will SIGTERM after inflight drops.
    let mut child = tokio::process::Command::new("/bin/sh")
        .args(["-c", "sleep 60"])
        .spawn()
        .unwrap();

    let counter = Arc::new(AtomicU64::new(2));
    let cfg = DrainConfig {
        max_request_duration: Duration::from_secs(1),
        drain_timeout: Duration::from_millis(0),
        extended_stream_drain: Duration::from_millis(0),
        sigterm_grace: Duration::from_millis(100),
    };

    // Decrement counter in background after a short virtual-time delay.
    let c = counter.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        c.store(0, Ordering::Relaxed);
    });

    drain_pipeline(&mut child, &cfg, counter.clone(), DrainReason::Eviction).await;
    // After drain, child should be terminated (wait should return immediately).
    let _ = child.wait().await;
}
