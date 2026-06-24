//! Periodic device VRAM/RAM sampler. Reads the shared device snapshot
//! at a fixed cadence and writes `device_samples` rows to the database
//! for historical charting.

use std::time::Duration;

use tokio::sync::watch;
use tracing::warn;

use crate::{db::Database, devices::snapshotter::SharedSnapshot};

/// Sampling cadence.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(10);

/// Spawn a task that writes device samples to the database every 10 seconds.
/// Runs until `shutdown` resolves to true.
pub fn spawn(
    db: Database,
    snapshot: SharedSnapshot,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(SAMPLE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        interval.tick().await; // skip the immediate first tick
        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        return;
                    }
                }
                _ = interval.tick() => {
                    let now = crate::tracking::now_unix_ms();
                    // Collect samples from the snapshot, then drop the
                    // read guard before awaiting DB writes (the guard
                    // is not Send and can't cross an await point).
                    let samples: Vec<(String, i64, i64)> = {
                        let snap = snapshot.read();
                        let mut v = Vec::new();
                        for g in &snap.gpus {
                            v.push((
                                format!("gpu:{}", g.id),
                                g.total_bytes as i64,
                                g.free_bytes as i64,
                            ));
                        }
                        if let Some(c) = &snap.cpu {
                            v.push((
                                "cpu".to_string(),
                                c.total_bytes as i64,
                                c.available_bytes as i64,
                            ));
                        }
                        v
                    };
                    for (device, total, free) in &samples {
                        if let Err(e) =
                            db.insert_device_sample(device, now, *total, *free).await
                        {
                            warn!(error = %e, device = %device, "failed to write device sample");
                        }
                    }
                }
            }
        }
    })
}
