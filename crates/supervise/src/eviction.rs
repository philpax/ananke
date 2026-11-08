//! [`RunLoop`]'s eviction planner: picking victims to drain when the packer
//! can't lay a service down against the current allocation table, and the
//! "yield to an active non-persistent peer" rule that keeps the persistent
//! watcher from sniping a peer mid-load.

use ananke_allocator::{
    AllocationTable,
    eviction::{EvictionCandidate, plan_eviction, plan_fixed_eviction, revalidate_eviction},
};
use ananke_config::validate::{DEFAULT_SERVICE_PRIORITY, Lifecycle};
use smol_str::SmolStr;
use tracing::{info, warn};

use crate::{
    RunLoop, SupervisorDeps,
    ensure::{ReservationFailure, RetryPackFailure},
    handle::EnsureSource,
    state::ServiceState,
};

pub fn eviction_candidates(
    exclude: &SmolStr,
    deps: &SupervisorDeps,
    table: &AllocationTable,
) -> Vec<EvictionCandidate> {
    let effective = deps.config.effective();
    let now = tokio::time::Instant::now();
    let mut candidates = Vec::new();
    for (_, handle) in deps.registry.all() {
        if &handle.name == exclude {
            continue;
        }
        let bytes = table
            .get(&handle.name)
            .map(|row| {
                row.values().fold(0u64, |sum, mb| {
                    sum.saturating_add(mb.saturating_mul(1024 * 1024))
                })
            })
            .unwrap_or(0);
        if bytes == 0 {
            continue;
        }
        let svc = effective
            .services
            .iter()
            .find(|svc| svc.name == handle.name);
        let priority = svc.map_or(DEFAULT_SERVICE_PRIORITY, |svc| svc.priority);
        let lifecycle = svc.map_or(Lifecycle::OnDemand, |svc| svc.lifecycle);
        // Dynamic services remain elastic despite long-lived UI connections.
        let dynamic = svc.is_some_and(|svc| {
            matches!(
                svc.allocation_mode,
                ananke_config::AllocationMode::Dynamic { .. }
            )
        });
        let settled = matches!(
            handle.peek_state(),
            ServiceState::Idle | ServiceState::Running
        );
        candidates.push(EvictionCandidate {
            name: handle.name.clone(),
            priority,
            lifecycle,
            idle: settled && (dynamic || deps.inflight.current(&handle.name) == 0),
            allocation_bytes: bytes,
        });
    }
    candidates.sort_by_key(|candidate| {
        std::cmp::Reverse(
            deps.activity
                .last(&candidate.name)
                .map(|last| now.saturating_duration_since(last).as_millis())
                .unwrap_or(u128::MAX),
        )
    });
    candidates
}

impl RunLoop {
    /// Pack failed against the current allocation table. Walk evictable peers
    /// in least-recently-used order (oldest activity first), adding one at
    /// a time until a strict pack succeeds with bounded reclaim credit. The
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
        let candidates = self.collect_eviction_candidates().await;
        let my_priority = self.current_svc().priority;
        let my_lifecycle = self.current_svc().lifecycle;

        let ranked: Vec<_> = candidates
            .iter()
            .filter(|candidate| candidate.is_evictable_by(my_priority, my_lifecycle))
            .collect();

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

        let svc = self.current_svc();
        let plan = plan_eviction(
            snap,
            table,
            &candidates,
            &svc,
            |post_snapshot, remaining| {
                let want = self.compute_reservation_map(post_snapshot, remaining)?;
                ananke_allocator::can_fit(&want, post_snapshot, remaining, Some(&svc.name))
                    .map_err(|err| {
                        ReservationFailure::PackFailed(
                            ananke_allocator::placement::PackError::WeightsDoNotFit {
                                shortfalls: vec![ananke_allocator::placement::DeviceShortfall {
                                    device: ananke_devices::DeviceId::from_slot(&err.slot),
                                    requested_bytes: err.needed_bytes,
                                    available_bytes: err.available_bytes,
                                }],
                            },
                        )
                    })?;
                Ok::<_, ReservationFailure>(want)
            },
        );
        let (want, victims) = match plan {
            Ok(plan) => plan,
            Err(err) => {
                let _ = self.packed_for_spawn.take();
                let reason = err.message();
                warn!(service = %svc.name, %reason, "eviction cannot provide enough capacity");
                return Err(RetryPackFailure::NotPossible(reason));
            }
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
        let candidates = self.collect_eviction_candidates().await;

        let reservations_now = self.deps.allocations.lock().clone();
        let snapshot = self.deps.snapshot.read().clone();
        let svc = self.current_svc();
        let mut ranked = candidates.clone();
        ranked.sort_by(|a, b| {
            b.idle
                .cmp(&a.idle)
                .then(a.priority.cmp(&b.priority))
                .then(a.allocation_bytes.cmp(&b.allocation_bytes))
        });
        let plan = plan_fixed_eviction(&snapshot, &reservations_now, &ranked, &svc, want);
        let to_evict = match plan {
            Ok(victims) if victims.is_empty() => return Ok(()),
            Ok(victims) => victims,
            Err(_) => Vec::new(),
        };

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

        // Drain acknowledgements can remove victim rows before the device snapshot refreshes.
        // Keep credit tied to the original snapshot, but recheck competing pledges.
        let current_table = self.deps.allocations.lock().clone();
        if let Err(again) = revalidate_eviction(
            &snapshot,
            &reservations_now,
            &current_table,
            &to_evict,
            want,
            Some(&self.init.identity.name),
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
    /// snapshotting our own supervisor, and skips peers with no allocation
    /// — an empty slot in the registry has nothing to give up, so it isn't
    /// an eviction candidate, it's just a registered service that hasn't
    /// started yet.
    pub(crate) async fn collect_eviction_candidates(&self) -> Vec<EvictionCandidate> {
        let table = self.deps.allocations.lock().clone();
        eviction_candidates(&self.init.identity.name, &self.deps, &table)
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
