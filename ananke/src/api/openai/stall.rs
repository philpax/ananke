//! Time-to-first-token stall detection for proxied OpenAI requests.
//!
//! A wedged llama.cpp child can accept a request — logging the chat format and
//! launching the slot — then never emit a single token, holding the HTTP
//! response open indefinitely. The process stays alive, so the crash path
//! never fires and the error-rate watchdog (which only sees *completed*
//! requests) never sees it either; from the daemon's side the request is
//! simply in-flight forever.
//!
//! This module arms a per-request timer when the request is forwarded and
//! signals the supervisor to auto-restart the service if no response frame
//! arrives within the configured timeout. The first frame from upstream
//! disarms it — proof the child is producing output.

use std::{sync::Arc, time::Duration};

use tokio::sync::oneshot;
use tracing::info;

use crate::{supervise::SupervisorHandle, tracking::progress::ProgressCell};

/// Disarm handle for a live stall timer, held by the proxied response body.
/// Calling [`Self::disarm`] on the first upstream frame cancels the pending
/// restart. Dropping it *without* disarming — e.g. the client disconnects
/// before any token — deliberately does not cancel the timer: a request that
/// stayed in-flight for the full timeout with no output is the wedge
/// signature regardless of who let go of the connection first.
pub struct StallDisarm(Option<oneshot::Sender<()>>);

impl StallDisarm {
    /// Record that the first response frame arrived, cancelling the pending
    /// restart. Idempotent: only the first call carries the signal.
    pub fn disarm(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}

/// Arm a stall timer for a single proxied request. Spawns a background task
/// that asks the supervisor to restart `run_id` if the returned
/// [`StallDisarm`] is not signalled within `timeout` **and** the service as a
/// whole has produced no response frame within that window (`progress` is the
/// per-service last-frame timestamp from `ProgressTable`).
///
/// The two-part condition is the crux: the per-request timer is only a
/// *sensor*. A request can sit past its timeout purely because it is queued
/// behind a healthy long-running generation (llama.cpp serialises requests at
/// `--parallel 1`), which produces frames for the *other* request the whole
/// time. Restarting then would kill healthy work. The run-level `progress`
/// check is the *verdict*: fire only if nothing, anywhere on this service, has
/// made progress for the full window — the actual wedge signature.
pub fn arm(
    handle: Arc<SupervisorHandle>,
    run_id: i64,
    timeout: Duration,
    progress: ProgressCell,
) -> StallDisarm {
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        tokio::select! {
            _ = tokio::time::sleep(timeout) => {
                if progress.idle_for() >= timeout {
                    handle.watchdog_stall(run_id);
                } else {
                    // The service produced a frame within the window — this
                    // request was merely queued behind healthy work, not
                    // wedged. Leave the run alone.
                    info!(
                        run_id,
                        "stall timer expired but service is producing frames; not restarting"
                    );
                }
            }
            // A first frame was forwarded for THIS request: the child is alive.
            // If instead the sender is dropped without a signal (`Err`), the
            // pattern does not match, tokio disables this branch, and the
            // timeout still runs to completion — so a client that disconnects
            // mid-wait still lets the run-level check decide.
            Ok(()) = rx => {}
        }
    });
    StallDisarm(Some(tx))
}
