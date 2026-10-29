//! [`RunLoop`]'s terminal-ish phases: `Failed` (retry backoff before the
//! next spawn attempt, or promotion to `Disabled`) and `Disabled` (parked
//! until an operator shuts down or re-enables the service).

use std::time::Duration;

use tracing::info;

use crate::{
    RunLoop, Step,
    ensure::EnsureFailure,
    handle::{DisableResult, EnableResult, EnsureResponse, SupervisorCommand},
    state::{DisableReason, Event as StateEvent, ServiceState, transition},
};

/// Backoff schedule for consecutive start failures before transitioning to
/// Disabled. Indexed by `retry_count` (0-based).
const FAILED_RETRY_BACKOFFS: [Duration; 3] = [
    Duration::from_secs(2),
    Duration::from_secs(5),
    Duration::from_secs(15),
];

impl RunLoop {
    pub(crate) async fn handle_failed(&mut self, retry_count: u8) -> Step {
        let idx = (retry_count as usize).min(FAILED_RETRY_BACKOFFS.len() - 1);
        let delay = FAILED_RETRY_BACKOFFS[idx];
        tokio::select! {
            _ = tokio::time::sleep(delay) => {
                // `handle_failed` is only reached in the `Failed` state, for which
                // `RetryAfterBackoff` is always defined — either bumping retry_count
                // or promoting to Disabled at the cap.
                let next = transition(&self.read_state(), StateEvent::RetryAfterBackoff);
                let next = if !matches!(next, ServiceState::Disabled { .. }) {
                    // Move back to Idle so the next Ensure triggers a fresh start.
                    ServiceState::Idle
                } else {
                    next
                };
                self.set_state(next);
                Step::Continue
            }
            cmd = self.rx.recv() => {
                match cmd {
                    Some(SupervisorCommand::Shutdown { ack }) => {
                        let _ = ack.send(());
                        Step::Exit
                    }
                    Some(SupervisorCommand::Ensure { ack, .. }) => {
                        // Surface a meaningful failure instead of dropping
                        // the ack: the client would otherwise see
                        // "supervisor unreachable", which doesn't tell them
                        // the service is in retry backoff.
                        let _ = ack.send(EnsureResponse::Unavailable(
                            EnsureFailure::StartFailed(format!(
                                "service {} is in Failed state; awaiting retry backoff",
                                self.init.identity.name
                            )),
                        ));
                        Step::Continue
                    }
                    // Failed means there's no child and no allocation to
                    // release, so drain/kill are instant no-ops but must
                    // still ack so the caller's `begin_drain.await` returns.
                    Some(SupervisorCommand::BeginDrain { ack, .. })
                    | Some(SupervisorCommand::FastKill { ack, .. }) => {
                        let _ = ack.send(());
                        Step::Continue
                    }
                    Some(SupervisorCommand::ActivityPing) => Step::Continue,
                    // No running child; a stall report is stale.
                    Some(SupervisorCommand::WatchdogStall { .. }) => Step::Continue,
                    Some(SupervisorCommand::Enable { ack }) => {
                        // Failed is not disabled; only clear a stale stored mark.
                        let _ = ack.send(match self.persist_operator_disable(false).await {
                            Ok(()) => EnableResult::NotDisabled,
                            Err(reason) => EnableResult::PersistFailed(reason),
                        });
                        Step::Continue
                    }
                    Some(SupervisorCommand::Disable { ack }) => {
                        if let Err(reason) = self.persist_operator_disable(true).await {
                            let _ = ack.send(DisableResult::PersistFailed(reason));
                            return Step::Continue;
                        }
                        // Disable a failed service: skip the retry and go to Disabled.
                        self.set_state(ServiceState::Disabled {
                            reason: DisableReason::UserDisabled,
                        });
                        let _ = ack.send(DisableResult::Disabled);
                        Step::Continue
                    }
                    None => Step::Exit,
                }
            }
        }
    }

    pub(crate) async fn handle_disabled(&mut self) -> Step {
        info!(service = %self.init.identity.name, "disabled; awaiting shutdown or enable");
        loop {
            match self.rx.recv().await {
                Some(SupervisorCommand::Shutdown { ack }) => {
                    let _ = ack.send(());
                    return Step::Exit;
                }
                Some(SupervisorCommand::Ensure { ack, .. }) => {
                    let _ = ack.send(EnsureResponse::Unavailable(EnsureFailure::ServiceDisabled(
                        "service disabled".into(),
                    )));
                }
                Some(SupervisorCommand::ActivityPing) => {}
                // Disabled: no running child, so a stall report is stale.
                Some(SupervisorCommand::WatchdogStall { .. }) => {}
                // Service is disabled; drain/kill are no-ops.
                Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::FastKill { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::Enable { ack }) => {
                    if let Err(reason) = self.persist_operator_disable(false).await {
                        let _ = ack.send(EnableResult::PersistFailed(reason));
                        continue;
                    }
                    // Transition back to Idle so the next Ensure can start it.
                    // Clear the auto-restart flap history: a manual re-enable is
                    // an operator override that grants a fresh restart budget.
                    // Without this, a service disabled with `AutoRestartLoop`
                    // (whose history is full by construction, all within the
                    // flap window) would be re-disabled on its very first
                    // error-rate trip after re-enable.
                    self.auto_restart_history.clear();
                    let next = transition(&self.read_state(), StateEvent::UserEnable);
                    self.set_state(next);
                    let _ = ack.send(EnableResult::Enabled);
                    return Step::Continue;
                }
                Some(SupervisorCommand::Disable { ack }) => {
                    // Already disabled. Store it anyway: an automatic disable
                    // (OOM, restart loop) the operator confirms should stick,
                    // and from then on it is the operator's, so the reason
                    // says so, as it would after a restart.
                    if let Err(reason) = self.persist_operator_disable(true).await {
                        let _ = ack.send(DisableResult::PersistFailed(reason));
                        continue;
                    }
                    if !matches!(
                        self.read_state(),
                        ServiceState::Disabled {
                            reason: DisableReason::UserDisabled
                        }
                    ) {
                        self.set_state(ServiceState::Disabled {
                            reason: DisableReason::UserDisabled,
                        });
                    }
                    let _ = ack.send(DisableResult::AlreadyDisabled);
                }
                None => return Step::Exit,
            }
        }
    }
}
