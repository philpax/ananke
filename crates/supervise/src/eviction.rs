//! [`RunLoop`]'s eviction planner: picking victims to drain when the packer
//! can't lay a service down against the current allocation table, and the
//! "yield to an active non-persistent peer" rule that keeps the persistent
//! watcher from sniping a peer mid-load.

use ananke_config::validate::{DEFAULT_SERVICE_PRIORITY, Lifecycle};
use tracing::{info, warn};

use crate::{
    RunLoop,
    ensure::{ReservationFailure, RetryPackFailure},
    handle::EnsureSource,
    state::ServiceState,
};

impl RunLoop {
    /// Pack failed against the current allocation table. Walk evictable peers
    /// in least-recently-used order (oldest activity first), adding one at
    /// a time to the victim set until the optimistic pack succeeds. The
    /// minimum number of LRU-first peers we need to drain, no more — a busy
    /// service whose cold peers already cover the demand never gets touched.
    ///
    /// Choosing LRU over "single biggest victim" is deliberate: among idle
    /// peers the disruption cost is near-zero (no one is using them), so
    /// evicting two cold services to preserve one warm one is the right
    /// trade. If the warm service happens to be the one with the most
    /// VRAM, we leave it alone and evict multiple LRU peers to cover the
    /// deficit.
    ///
    /// (Going through `try_eviction_to_fit` here instead wouldn't work:
    /// `collect_eviction_candidates` reads peer state through the mirror,
    /// which is fine, but that helper is sized for the
    /// "fit-but-missing-headroom" case and doesn't re-run the packer after
    /// each eviction; this one does.)
    pub(crate) async fn retry_pack_with_eviction(
        &mut self,
        snap: &ananke_devices::DeviceSnapshot,
        table: &ananke_allocator::AllocationTable,
    ) -> Result<
        (
            std::collections::BTreeMap<ananke_config::DeviceSlot, u64>,
            Vec<smol_str::SmolStr>,
        ),
        RetryPackFailure,
    > {
        let candidates = self.collect_eviction_candidates();
        let my_priority = self.current_svc().priority;
        let my_lifecycle = self.current_svc().lifecycle;

        // LRU-first ordering over evictable peers. `collect_eviction_candidates`
        // already excludes zero-allocation peers, so everything in `candidates`
        // holds real VRAM. Services that have never been pinged sort as "most
        // LRU" (effectively infinite age), so an idle peer is a first-class
        // victim candidate — except for tied-priority persistent peers when
        // the requester is on-demand, which `is_evictable_by` filters out.
        let now = tokio::time::Instant::now();
        let mut ranked: Vec<(&ananke_allocator::eviction::EvictionCandidate, u128)> = candidates
            .iter()
            .filter(|c| c.is_evictable_by(my_priority, my_lifecycle))
            .map(|c| {
                let age_ms = self
                    .deps
                    .activity
                    .last(&c.name)
                    .map(|t| now.saturating_duration_since(t).as_millis())
                    .unwrap_or(u128::MAX);
                (c, age_ms)
            })
            .collect();
        ranked.sort_by_key(|(_, age)| std::cmp::Reverse(*age));

        if ranked.is_empty() {
            // No peer is currently evictable. Distinguish two cases so the
            // caller can queue vs reject appropriately:
            //   - there IS a busy peer at our priority or lower, so waiting
            //     for it to idle would make it evictable → `WaitForBusy`
            //     (returning the set of peers to watch so the queue loop
            //     can skip expensive retries while they're all still busy).
            //   - all peers are higher priority, or there are none → hard
            //     reject with the pack reason.
            let busy_peers: Vec<smol_str::SmolStr> = candidates
                .iter()
                .filter(|c| !c.idle && c.priority <= my_priority)
                .map(|c| c.name.clone())
                .collect();
            let _ = self.packed_for_spawn.take();
            if !busy_peers.is_empty() {
                info!(
                    service = %self.init.identity.name,
                    busy_peers = ?busy_peers,
                    my_priority,
                    "no evictable candidates yet; waiting for busy peer to idle"
                );
                return Err(RetryPackFailure::WaitForBusy { busy_peers });
            }
            info!(
                service = %self.init.identity.name,
                candidates = candidates.len(),
                my_priority,
                "eviction selection failed: no evictable candidates for placement"
            );
            let reason = self
                .compute_reservation_map(snap, table)
                .err()
                .map(ReservationFailure::message)
                .unwrap_or_else(|| "placement: unknown failure".into());
            return Err(RetryPackFailure::NotPossible(reason));
        }

        // Greedy LRU-first fill: walk peers in LRU order, extending the
        // victim set one at a time, and re-run the optimistic packer after
        // every addition. Stop as soon as the pack succeeds — that's the
        // minimum LRU-first victim set that covers the layout.
        let mut victims: Vec<smol_str::SmolStr> = Vec::new();
        let mut winning_want: Option<std::collections::BTreeMap<ananke_config::DeviceSlot, u64>> =
            None;
        for (cand, _) in &ranked {
            victims.push(cand.name.clone());
            let mut filtered = table.clone();
            for v in &victims {
                filtered.remove(v);
            }
            if let Ok(w) = self.compute_reservation_map_optimistic(snap, &filtered) {
                winning_want = Some(w);
                break;
            }
        }

        let Some(want) = winning_want else {
            // Even with every evictable peer treated as gone the packer
            // still can't lay out the model. Report the last pack error
            // so the operator sees the actual deficit (which GPU, what
            // bytes), not a generic "no fit".
            let mut filtered = table.clone();
            for v in &victims {
                filtered.remove(v);
            }
            let reason = self
                .compute_reservation_map_optimistic(snap, &filtered)
                .err()
                .map(ReservationFailure::message)
                .unwrap_or_else(|| "placement: unknown failure".into());
            warn!(
                service = %self.init.identity.name,
                reason = %reason,
                evictable_count = victims.len(),
                "optimistic pack failed even with all evictable peers treated as gone"
            );
            return Err(RetryPackFailure::NotPossible(reason));
        };

        info!(
            service = %self.init.identity.name,
            evict_count = victims.len(),
            victims = ?victims,
            evictable_considered = ranked.len(),
            "eviction planned (LRU-first minimum victim set)"
        );
        for victim in &victims {
            if let Some(handle) = self.deps.registry.get(victim) {
                handle
                    .begin_drain(crate::drain::DrainReason::Eviction)
                    .await;
            }
        }
        Ok((want, victims))
    }

    /// Try to make room by draining lower-priority services, then re-check
    /// `can_fit`. `Ok(())` means the caller can proceed to reserve;
    /// `Err(RetryPackFailure::WaitForBusy)` means eviction couldn't pick a
    /// victim right now but a busy same-or-lower-priority peer exists, so
    /// the caller should queue and retry; `Err(RetryPackFailure::NotPossible)`
    /// means no amount of waiting will help.
    pub(crate) async fn try_eviction_to_fit(
        &mut self,
        want: &std::collections::BTreeMap<ananke_config::DeviceSlot, u64>,
        nofit: &ananke_allocator::NoFit,
        source: EnsureSource,
    ) -> Result<(), RetryPackFailure> {
        // Matching the check in `handle_idle_ensure` and
        // `retry_queued_ensure`: a background-watcher-driven persistent
        // ensure yields when a non-persistent peer is actively loading or
        // running. User-driven ensures may proceed to eviction.
        if self.should_yield_to_active_nonpersistent(source) {
            info!(
                service = %self.init.identity.name,
                "persistent ensure yielding to active non-persistent peer (post-pack fit)"
            );
            return Err(RetryPackFailure::NotPossible(
                "persistent service yielding to active non-persistent peer".into(),
            ));
        }
        let candidates = self.collect_eviction_candidates();

        let reservations_now = self.deps.allocations.lock().clone();
        // `nofit.available_bytes` is already reservation-adjusted
        // (`min(snap.free, snap.total - sum_of_reservations)`), so pass
        // it through verbatim. Using `snap.free_bytes` here would let
        // running services' pledges hide behind the raw snapshot free
        // and short-circuit eviction — exactly what happens in the
        // fake-spawner harness where the snapshot doesn't move as
        // services start up.
        let to_evict = ananke_allocator::eviction::select_for_slot(
            nofit.needed_bytes,
            &nofit.slot,
            self.current_svc().priority,
            self.current_svc().lifecycle,
            &candidates,
            &reservations_now,
            nofit.available_bytes,
        );

        if to_evict.is_empty() {
            // Same distinction as `retry_pack_with_eviction`: is there a busy
            // peer we could evict once it idles, or are we genuinely stuck?
            let my_priority = self.current_svc().priority;
            let busy_peers: Vec<smol_str::SmolStr> = candidates
                .iter()
                .filter(|c| !c.idle && c.priority <= my_priority)
                .map(|c| c.name.clone())
                .collect();
            let _ = self.packed_for_spawn.take();
            if !busy_peers.is_empty() {
                info!(
                    service = %self.init.identity.name,
                    busy_peers = ?busy_peers,
                    needed_bytes = nofit.needed_bytes,
                    available_bytes = nofit.available_bytes,
                    slot = ?nofit.slot,
                    "no evictable candidates yet; waiting for busy peer to idle"
                );
                return Err(RetryPackFailure::WaitForBusy { busy_peers });
            }
            info!(
                service = %self.init.identity.name,
                candidates = candidates.len(),
                needed_bytes = nofit.needed_bytes,
                available_bytes = nofit.available_bytes,
                slot = ?nofit.slot,
                "eviction selection failed: no evictable candidates cover the deficit"
            );
            return Err(RetryPackFailure::NotPossible(format!("{nofit}")));
        }

        info!(
            service = %self.init.identity.name,
            evict_count = to_evict.len(),
            victims = ?to_evict,
            "eviction planned (fit feasible after drain)"
        );
        for victim in &to_evict {
            if let Some(handle) = self.deps.registry.get(victim) {
                handle
                    .begin_drain(crate::drain::DrainReason::Eviction)
                    .await;
            }
        }

        // Re-attempt feasibility after evictions. `begin_drain` is
        // non-blocking — the victims' allocation rows aren't actually removed
        // until each drain finishes (see `record_drain_complete`), so the raw
        // allocation table still reflects them at this moment.
        // `can_fit_after_eviction` treats the planned evictees as already
        // freed; if the drains later fail or stall, the child spawn will see
        // the real state and OOM-retry will handle it.
        let snap2 = self.deps.snapshot.read().clone();
        let table2 = self.deps.allocations.lock().clone();
        if let Err(again) = ananke_allocator::can_fit_after_eviction(
            want,
            &snap2,
            &table2,
            Some(&self.init.identity.name),
            &to_evict,
        ) {
            let _ = self.packed_for_spawn.take();
            return Err(RetryPackFailure::NotPossible(format!(
                "eviction insufficient: {again}"
            )));
        }
        Ok(())
    }

    /// Enumerate every other service that currently holds a VRAM reservation
    /// and could plausibly be displaced. Skips self so we don't deadlock
    /// snapshotting our own supervisor; see [`eviction_candidates`].
    pub(crate) fn collect_eviction_candidates(
        &self,
    ) -> Vec<ananke_allocator::eviction::EvictionCandidate> {
        eviction_candidates(&self.deps, Some(self.init.identity.name.as_str()))
    }

    /// The "persistent yields to active non-persistent" predicate: this
    /// service is `Persistent` and at least one peer service with
    /// `Lifecycle::OnDemand` is currently in `Starting` or `Running`.
    ///
    /// Callers use this to stand down from an eviction-requiring start
    /// rather than snipe a peer that's in the middle of loading or
    /// running. The persistent watcher will retry on its own cadence
    /// when the pool quiets, so there's no reclamation-deadline pressure.
    pub(crate) fn should_yield_to_active_nonpersistent(&self, source: EnsureSource) -> bool {
        // User-driven requests are allowed to evict idle on-demand peers
        // regardless of their running state; only background watcher re-ensures
        // should stand down when a non-persistent peer is active.
        if source == EnsureSource::UserRequest {
            return false;
        }
        if self.current_svc().lifecycle != ananke_config::Lifecycle::Persistent {
            return false;
        }
        let lifecycle_by_name: std::collections::BTreeMap<_, _> = {
            let eff = self.deps.config.effective();
            eff.services
                .iter()
                .map(|s| (s.name.clone(), s.lifecycle))
                .collect()
        };
        for (_, handle) in self.deps.registry.all() {
            if handle.name.as_str() == self.init.identity.name.as_str() {
                continue;
            }
            let lifecycle = lifecycle_by_name
                .get(&handle.name)
                .copied()
                .unwrap_or(ananke_config::Lifecycle::OnDemand);
            if lifecycle == ananke_config::Lifecycle::Persistent {
                continue;
            }
            if matches!(
                handle.peek_state(),
                ServiceState::Starting | ServiceState::Running,
            ) {
                return true;
            }
        }
        false
    }

    /// True if a named peer is currently "still busy" for the purpose of
    /// the queued-ensure retry precheck. A peer counts as busy when it
    /// has in-flight work OR is in `Starting` (loading, not yet eligible
    /// for eviction). Without the `Starting`-aware branch the 250 ms tick
    /// would re-run the full estimator + packer every tick while a peer
    /// loads, because `inflight == 0` during startup.
    pub(crate) fn peer_still_busy_for_precheck(&self, name: &smol_str::SmolStr) -> bool {
        if self.deps.inflight.current(name) > 0 {
            return true;
        }
        self.deps
            .registry
            .get(name.as_str())
            .map(|h| matches!(h.peek_state(), ServiceState::Starting))
            .unwrap_or(false)
    }
}

/// Every registered service except `exclude` that holds an allocation, as
/// eviction candidates. Peers with no allocation are skipped: an empty slot
/// in the registry has nothing to give up, so it isn't an eviction
/// candidate, it's just a registered service that hasn't started yet.
/// Shared by service starts and manual reservations, so both evict by the
/// same rules.
pub fn eviction_candidates(
    deps: &crate::SupervisorDeps,
    exclude: Option<&str>,
) -> Vec<ananke_allocator::eviction::EvictionCandidate> {
    let all_services = deps.registry.all();
    // Materialise the priority + lifecycle + dynamic-mode flag from the
    // live config up front, so the arc-swap guard is released promptly. The dynamic-mode bit is what lets the planner
    // treat balloon services as elastic — see the idle-computation
    // comment below.
    let svc_meta_by_name: std::collections::BTreeMap<_, _> = {
        let eff = deps.config.effective();
        eff.services
            .iter()
            .map(|s| {
                let is_dynamic = matches!(
                    s.allocation_mode,
                    ananke_config::AllocationMode::Dynamic { .. }
                );
                (s.name.clone(), (s.priority, s.lifecycle, is_dynamic))
            })
            .collect()
    };
    let mut out = Vec::new();
    for (_name, handle) in all_services {
        if exclude.is_some_and(|name| handle.name.as_str() == name) {
            continue;
        }
        let alloc_mb = deps
            .allocations
            .lock()
            .get(&handle.name)
            .cloned()
            .unwrap_or_default();
        let bytes = alloc_mb.values().sum::<u64>() * 1024 * 1024;
        if bytes == 0 {
            continue;
        }
        // `peek_state` reads the shared mirror under a parking_lot
        // mutex — no mailbox hop, no circular wait. When this is
        // called from inside `handle_idle_ensure` the peer supervisor
        // may be mid-drain and unable to service commands; reading the
        // mirror directly is the only safe path.
        let state = handle.peek_state();
        let (priority, lifecycle, is_dynamic) = svc_meta_by_name
            .get(&handle.name)
            .copied()
            .unwrap_or((DEFAULT_SERVICE_PRIORITY, Lifecycle::OnDemand, false));
        // "Idle" for eviction purposes means "no user-facing work
        // in flight on a settled supervisor" — either literally
        // Idle (not running) or Running with no in-flight requests.
        // Starting is excluded: the child is spawned but not yet
        // healthy and its start-bus still holds queued callers
        // who'd all fail if we tore it down.
        //
        // Dynamic-allocation services are the explicit exception. By
        // choosing `allocation.mode = "dynamic"` the operator has
        // declared the service elastic — happy to be torn down when
        // a peer needs VRAM. The most common case is ComfyUI: its
        // web UI keeps a long-lived `/ws` open for live updates, so
        // its inflight counter is rarely zero even when no image is
        // actually generating. Treating that idle-with-open-UI as
        // "busy" deadlocks tied-priority on-demand peers (chat hits
        // 503 / silent queue) while ComfyUI's 7 GiB pledge sits
        // unused. Funnelling dynamic services into the idle bucket
        // restores the eviction path the operator opted into.
        let in_flight = deps.inflight.current(&handle.name);
        let settled = matches!(state, ServiceState::Idle | ServiceState::Running);
        let idle = settled && (is_dynamic || in_flight == 0);
        out.push(ananke_allocator::eviction::EvictionCandidate {
            name: handle.name.clone(),
            priority,
            lifecycle,
            idle,
            allocation_bytes: bytes,
        });
    }
    out
}
