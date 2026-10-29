//! [`RunLoop`]'s `Idle` phase: waiting for an `Ensure`, computing the
//! reservation for the next spawn, and — when the fit is blocked by a busy
//! peer — parking the caller on a queue that the poll-tick branch retries.

use std::time::Duration;

use tracing::{info, warn};

use crate::{
    RunLoop, Step,
    ensure::{EnsureFailure, ReservationFailure, RetryPackFailure},
    handle::{
        DisableResult, EnableResult, EnsureResponse, EnsureSource, StartFailure, StartFailureKind,
        StartOutcome, SupervisorCommand,
    },
    state::{DisableReason, Event as StateEvent, ServiceState, transition},
};

/// Poll interval while a supervisor is queued behind a busy peer waiting
/// for it to idle. Each tick is a cheap atomic-load precheck against the
/// watched peers' inflight counters (see `retry_queued_ensure`); only when
/// a peer has actually gone idle do we run the full estimator + packer.
/// 250 ms is fast enough that a queued request wakes up within a quarter
/// second of the peer finishing its response, while keeping the tick
/// noise low enough that the logs aren't drowned when a queued ensure
/// is waiting for a peer that's loading.
const QUEUE_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Hard upper bound on how long an Ensure may sit in the start queue
/// waiting for a busy peer to idle. Past this point the queue gives up
/// and resolves with [`StartFailureKind::Blocked`] so the client sees a
/// 503 with a clear "blocked by peer X" message instead of waiting the
/// full `max_request_duration_ms` (default 10 min) and silently 503'ing
/// with "start timed out".
///
/// Sized to absorb the brief "peer just finishing its response" window
/// (the happy-path queue test releases at 400 ms; a multi-token
/// streaming completion that's about to end fits comfortably) without
/// silently absorbing the genuinely-stuck case. Past 10 s the
/// expectation is the client should see a structured error and decide
/// whether to wait, kill the blocker, or try a different model. The
/// blocker is, by definition, a non-elastic peer at our priority or
/// higher — dynamic-allocation services are always evictable (see
/// `collect_eviction_candidates`), so this timeout exists for the
/// genuinely-stuck case where waiting longer would not help.
const QUEUE_BLOCKED_GRACE: Duration = Duration::from_secs(10);

impl RunLoop {
    /// For on_demand services we wait here for an Ensure. Persistent services
    /// have the daemon call ensure() synthetically at boot.
    ///
    /// The poll-tick branch of the select is only armed while a queued Ensure
    /// is parked on `pending_ensure_bus` (busy peer waiting to idle); when no
    /// queue is pending, the loop is a pure command-recv.
    pub(crate) async fn handle_idle(&mut self) -> Step {
        // An on-request periodic restart drained to Idle while holding the
        // triggering request's Ensure. Replay it through the normal
        // idle-ensure path now so the caller blocks on the fresh spawn (a
        // Waiting response on a real start bus) rather than getting
        // AlreadyRunning against a child that no longer exists.
        if let Some((ack, source)) = self.deferred_ensure.take()
            && self.handle_idle_ensure(ack, source).await
        {
            return Step::Continue;
        }
        loop {
            tokio::select! {
                _ = tokio::time::sleep(QUEUE_POLL_INTERVAL), if self.pending_ensure_bus.is_some() => {
                    if self.retry_queued_ensure().await {
                        return Step::Continue;
                    }
                }
                cmd = self.rx.recv() => {
                    match cmd {
                        Some(SupervisorCommand::Shutdown { ack }) => {
                            self.fail_queue(
                                StartFailureKind::Disabled,
                                "supervisor shutting down".into(),
                            );
                            let _ = ack.send(());
                            return Step::Exit;
                        }
                        Some(SupervisorCommand::Ensure { ack, source }) => {
                            // If there's already a queued Ensure, subscribe
                            // the new caller to the same bus (coalesce) up to
                            // `start_queue_depth`. Otherwise go through the
                            // normal fit path.
                            if let Some(sender) = self.pending_ensure_bus.as_ref() {
                                if sender.receiver_count() >= self.current_svc().start_queue_depth {
                                    let _ = ack.send(EnsureResponse::QueueFull);
                                } else {
                                    let rx = sender.subscribe();
                                    let _ = ack.send(EnsureResponse::Waiting { rx });
                                }
                            } else if self.handle_idle_ensure(ack, source).await {
                                return Step::Continue;
                            }
                        }
                        Some(SupervisorCommand::ActivityPing) => {}
                        // Not running: a stall can only be reported against a
                        // Running run, so this is always stale here.
                        Some(SupervisorCommand::WatchdogStall { .. }) => {}
                        // Service is not running; drain/kill commands are no-ops.
                        Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                            let _ = ack.send(());
                        }
                        Some(SupervisorCommand::FastKill { ack, .. }) => {
                            let _ = ack.send(());
                        }
                        Some(SupervisorCommand::Enable { ack }) => {
                            // Idle is already enabled; only clear a stale stored mark.
                            let _ = ack.send(match self.persist_operator_disable(false).await {
                                Ok(()) => EnableResult::NotDisabled,
                                Err(reason) => EnableResult::PersistFailed(reason),
                            });
                        }
                        Some(SupervisorCommand::Disable { ack }) => {
                            if let Err(reason) = self.persist_operator_disable(true).await {
                                let _ = ack.send(DisableResult::PersistFailed(reason));
                                continue;
                            }
                            // Transition idle service directly to Disabled.
                            self.fail_queue(
                                StartFailureKind::Disabled,
                                "service disabled by operator".into(),
                            );
                            self.set_state(ServiceState::Disabled {
                                reason: DisableReason::UserDisabled,
                            });
                            let _ = ack.send(DisableResult::Disabled);
                            return Step::Continue;
                        }
                        None => {
                            self.fail_queue(
                                StartFailureKind::Disabled,
                                "supervisor channel closed".into(),
                            );
                            return Step::Exit;
                        }
                    }
                }
            }
        }
    }

    /// Body of an `Ensure` received while Idle. Returns `true` if the
    /// reservation succeeded and the state has transitioned to Starting (so
    /// the Idle loop should break); `false` if the ensure was rejected and
    /// the loop should keep waiting for the next command.
    pub(crate) async fn handle_idle_ensure(
        &mut self,
        ack: tokio::sync::oneshot::Sender<EnsureResponse>,
        source: EnsureSource,
    ) -> bool {
        let snap = self.deps.snapshot.read().clone();
        let table = self.deps.allocations.lock().clone();

        let (want, pre_evicted) = match self.compute_reservation_map(&snap, &table) {
            Ok(w) => (w, Vec::new()),
            Err(ReservationFailure::PackFailed(msg)) => {
                // Pack couldn't lay the model down given current reservations
                // (e.g. an in-between layer didn't fit on any allowed GPU).
                // Retry with lower-priority services treated as evicted; if
                // pack succeeds, drain those victims and carry them through
                // to the feasibility check.
                //
                // Before attempting eviction, check the "persistent yields to
                // active non-persistent" rule: a persistent service about to
                // evict a peer stands down if any non-persistent peer is
                // Starting or Running — but only when the ensure originated
                // from the background watcher, not from a user request. A
                // user explicitly asking for a persistent service should be
                // allowed to evict idle on-demand peers.
                if self.should_yield_to_active_nonpersistent(source) {
                    info!(
                        service = %self.init.identity.name,
                        "persistent ensure yielding to active non-persistent peer"
                    );
                    let _ = ack.send(EnsureResponse::Unavailable(
                        EnsureFailure::InsufficientCapacity(
                            "persistent service yielding to active non-persistent peer".into(),
                        ),
                    ));
                    return false;
                }
                info!(
                    service = %self.init.identity.name,
                    reason = %msg,
                    "initial pack failed; retrying with eviction"
                );
                match self.retry_pack_with_eviction(&snap, &table).await {
                    Ok((want, victims)) => (want, victims),
                    Err(RetryPackFailure::NotPossible(retry_reason)) => {
                        // Reason is already logged by `retry_pack_with_eviction`
                        // (either "optimistic pack failed" or "no evictable
                        // candidates"), so no second log here — the consuming
                        // handler emits the client-facing line.
                        let _ = ack.send(EnsureResponse::Unavailable(
                            EnsureFailure::InsufficientCapacity(retry_reason),
                        ));
                        return false;
                    }
                    Err(RetryPackFailure::WaitForBusy { busy_peers }) => {
                        // Park the caller on a broadcast bus and stay in
                        // Idle. The Idle loop's poll-tick branch retries
                        // the ensure periodically via `retry_queued_ensure`;
                        // Shutdown/Disable/another Ensure that arrive
                        // while we're queued flow through the normal
                        // command-recv arm and drain the bus appropriately.
                        return self.enter_queue(ack, busy_peers, source);
                    }
                }
            }
            Err(other) => {
                let msg = other.message();
                warn!(
                    service = %self.init.identity.name,
                    reason = %msg,
                    "ensure failed: reservation computation error"
                );
                let _ = ack.send(EnsureResponse::Unavailable(EnsureFailure::ServiceDisabled(
                    msg,
                )));
                return false;
            }
        };

        // Feasibility check. When we came through `retry_pack_with_eviction`,
        // the victims it drained are still sitting in the allocation table
        // (drains are in-flight) and their supervisors are blocked executing
        // the drain — `try_eviction_to_fit` can't poll their state. Use
        // `can_fit_after_eviction` with the already-committed victim list so
        // we don't loop back through select_for_slot only to find zero
        // candidates.
        let fit_result = if pre_evicted.is_empty() {
            ananke_allocator::can_fit(&want, &snap, &table, Some(&self.init.identity.name))
        } else {
            ananke_allocator::can_fit_after_eviction(
                &want,
                &snap,
                &table,
                Some(&self.init.identity.name),
                &pre_evicted,
            )
        };
        info!(
            service = %self.init.identity.name,
            fit_ok = fit_result.is_ok(),
            pre_evicted = ?pre_evicted,
            "fit_result computed"
        );
        if let Err(nofit) = fit_result {
            match self.try_eviction_to_fit(&want, &nofit, source).await {
                Ok(()) => {}
                Err(RetryPackFailure::NotPossible(reason)) => {
                    let _ = ack.send(EnsureResponse::Unavailable(
                        EnsureFailure::InsufficientCapacity(reason),
                    ));
                    return false;
                }
                Err(RetryPackFailure::WaitForBusy { busy_peers }) => {
                    return self.enter_queue(ack, busy_peers, source);
                }
            }
        }

        // Reserve in the allocation table before spawning, capturing what the
        // rolling update will need when the service later drains back to Idle.
        self.capture_rolling_base();
        self.deps
            .allocations
            .lock()
            .insert(self.init.identity.name.clone(), want);
        self.emit_allocation_changed();

        // Create broadcast channel and subscribe the caller.
        let sender = tokio::sync::broadcast::channel::<StartOutcome>(16).0;
        let bus_rx = sender.subscribe();
        let _ = ack.send(EnsureResponse::Waiting { rx: bus_rx });
        self.start_bus_carry = Some(sender);

        let next = transition(&self.read_state(), StateEvent::SpawnRequested);
        self.set_state(next);
        true
    }

    /// Park an Ensure whose fit is blocked by a busy peer. Creates the shared
    /// broadcast bus, replies `Waiting` to the caller, records the busy peers
    /// for the cheap poll-tick precheck, and stashes the sender on
    /// `self.pending_ensure_bus` so the Idle loop's poll-tick branch can
    /// later retry the fit. Always returns `false` — the supervisor stays in
    /// Idle until either a retry succeeds or a command drains the bus.
    pub(crate) fn enter_queue(
        &mut self,
        ack: tokio::sync::oneshot::Sender<EnsureResponse>,
        busy_peers: Vec<smol_str::SmolStr>,
        source: EnsureSource,
    ) -> bool {
        let sender = tokio::sync::broadcast::channel::<StartOutcome>(16).0;
        let bus_rx = sender.subscribe();
        let _ = ack.send(EnsureResponse::Waiting { rx: bus_rx });
        self.pending_ensure_bus = Some(sender);
        self.pending_ensure_source = source;
        self.queued_watch = busy_peers;
        // Stamp the first time this Ensure parks in the queue so
        // QUEUE_BLOCKED_GRACE can fire even if `queued_watch` keeps
        // shifting across retry ticks.
        self.queued_since = Some(tokio::time::Instant::now());
        false
    }

    /// Retry a queued Ensure. Called from the Idle loop's poll-tick when
    /// `pending_ensure_bus` is Some. On success, promotes the queued bus to
    /// `start_bus_carry` and transitions to Starting. On hard-fail, drains
    /// the bus with `Err` and clears the queue. On continued soft-wait,
    /// leaves the queue in place for the next tick.
    ///
    /// Cheap precheck up front: if every peer in `queued_watch` still has
    /// inflight > 0, nothing actionable has changed since last tick and we
    /// skip the expensive estimator + packer path entirely. Only when at
    /// least one watched peer has gone idle do we run the full retry. This
    /// is what keeps a 30-second wait from producing 60+ GGUF reads and
    /// 180+ info log lines.
    pub(crate) async fn retry_queued_ensure(&mut self) -> bool {
        // Bail out of the queue entirely once we've been parked here for
        // longer than `QUEUE_BLOCKED_GRACE`. Without this, a request
        // blocked by a tied-priority non-elastic peer (e.g. another
        // model mid-generation that's about to exceed the user's
        // patience) hangs all the way to `max_request_duration_ms` and
        // returns a generic "start timed out". With it, the client sees
        // a 503 + structured "blocked by peer X" within ~30 s. Dynamic
        // peers don't reach this branch because they are always
        // evictable (see `collect_eviction_candidates`).
        if let Some(since) = self.queued_since
            && since.elapsed() > QUEUE_BLOCKED_GRACE
        {
            let busy_peers = self.queued_watch.clone();
            let log_summary = if busy_peers.is_empty() {
                "unknown".to_string()
            } else {
                busy_peers
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            info!(
                service = %self.init.identity.name,
                busy_peers = %log_summary,
                grace_secs = QUEUE_BLOCKED_GRACE.as_secs(),
                "queue grace exceeded; failing queued ensure"
            );
            // The `message` field on `StartFailure` is a human-readable
            // log breadcrumb only; the wire layer renders
            // off the structured `busy_peers` list inside the kind
            // and ignores this string.
            let log_message = format!(
                "blocked by busy peer(s) for {:?}: {log_summary}",
                QUEUE_BLOCKED_GRACE
            );
            self.fail_queue(StartFailureKind::Blocked { busy_peers }, log_message);
            return false;
        }

        // Same yield rule as `handle_idle_ensure`: a background-watcher-driven
        // persistent ensure queued behind a busy peer stands down the moment
        // any non-persistent peer enters Starting/Running. Without this check,
        // the watcher would wait for the peer to finish loading and then
        // immediately evict it — the exact opposite of what it should do.
        // User-driven ensures are exempt: they may proceed to eviction.
        let source = self.pending_ensure_source;
        if self.should_yield_to_active_nonpersistent(source) {
            info!(
                service = %self.init.identity.name,
                "queued persistent ensure yielding to active non-persistent peer"
            );
            self.fail_queue(
                StartFailureKind::NoFit,
                "persistent service yielding to active non-persistent peer".into(),
            );
            return false;
        }

        if !self.queued_watch.is_empty()
            && self
                .queued_watch
                .iter()
                .all(|name| self.peer_still_busy_for_precheck(name))
        {
            return false;
        }

        let snap = self.deps.snapshot.read().clone();
        let table = self.deps.allocations.lock().clone();

        let (want, pre_evicted) = match self.compute_reservation_map(&snap, &table) {
            Ok(w) => (w, Vec::new()),
            Err(ReservationFailure::PackFailed(_)) => {
                match self.retry_pack_with_eviction(&snap, &table).await {
                    Ok((want, victims)) => (want, victims),
                    Err(RetryPackFailure::WaitForBusy { busy_peers }) => {
                        self.queued_watch = busy_peers;
                        return false;
                    }
                    Err(RetryPackFailure::NotPossible(reason)) => {
                        self.fail_queue(StartFailureKind::LaunchFailed, reason);
                        return false;
                    }
                }
            }
            Err(other) => {
                self.fail_queue(StartFailureKind::Disabled, other.message());
                return false;
            }
        };

        // Feasibility re-check on the post-eviction table, mirroring the
        // shape of `handle_idle_ensure`'s fit branch.
        let fit_result = if pre_evicted.is_empty() {
            ananke_allocator::can_fit(&want, &snap, &table, Some(&self.init.identity.name))
        } else {
            ananke_allocator::can_fit_after_eviction(
                &want,
                &snap,
                &table,
                Some(&self.init.identity.name),
                &pre_evicted,
            )
        };
        if let Err(nofit) = fit_result {
            match self.try_eviction_to_fit(&want, &nofit, source).await {
                Ok(()) => {}
                Err(RetryPackFailure::NotPossible(reason)) => {
                    self.fail_queue(StartFailureKind::LaunchFailed, reason);
                    return false;
                }
                // Still waiting for the busy peer; refresh the watch set
                // so next tick skips until something changes.
                Err(RetryPackFailure::WaitForBusy { busy_peers }) => {
                    self.queued_watch = busy_peers;
                    return false;
                }
            }
        }

        // Commit the reservation + promote the queued bus to the start
        // carry, then transition to Starting. `handle_active_lifecycle`
        // will pick it up from here.
        self.capture_rolling_base();
        self.deps
            .allocations
            .lock()
            .insert(self.init.identity.name.clone(), want);
        self.emit_allocation_changed();
        if let Some(sender) = self.pending_ensure_bus.take() {
            self.start_bus_carry = Some(sender);
        }
        self.queued_watch.clear();
        self.queued_since = None;
        let next = transition(&self.read_state(), StateEvent::SpawnRequested);
        self.set_state(next);
        true
    }

    /// Drain the queued Ensure bus with an error outcome and clear the
    /// pending state. Called on hard-reject, shutdown, disable, or any
    /// other terminal interrupt while queued.
    pub(crate) fn fail_queue(&mut self, kind: StartFailureKind, message: String) {
        self.queued_watch.clear();
        self.queued_since = None;
        if let Some(sender) = self.pending_ensure_bus.take() {
            let _ = sender.send(StartOutcome::Err(StartFailure { kind, message }));
        }
    }
}
