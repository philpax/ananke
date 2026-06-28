// Centralised store for services and devices. Subscribed to via
// useSyncExternalStore — no TanStack Query involved. The events
// WebSocket patches the store directly for instant UI updates; a
// debounced background refresh fills in fields the event payload
// doesn't carry (pid, run_id, free_bytes, …).

import { useEffect, useSyncExternalStore } from "react";

import { api, type DeviceSummary, type ServiceSummary } from "./client.ts";
import type { SystemEvent } from "./events.ts";

export type StoreResult<T> = {
  data: T | undefined;
  isPending: boolean;
  error: Error | null;
};

type SystemSnapshot = {
  services: ServiceSummary[];
  devices: DeviceSummary[];
  status: "loading" | "ready" | "error";
  error: Error | null;
  hasData: boolean;
};

// --- Module-level state ---

const INITIAL: SystemSnapshot = {
  services: [],
  devices: [],
  status: "loading",
  error: null,
  hasData: false,
};

let snapshot: SystemSnapshot = INITIAL;
const listeners = new Set<() => void>();

let fetchInFlight: Promise<void> | null = null;
let pollTimer: ReturnType<typeof setInterval> | null = null;
let debounceTimer: ReturnType<typeof setTimeout> | null = null;
let initStarted = false;

// --- Store primitives ---

function setSnapshot(next: SystemSnapshot): void {
  snapshot = next;
  for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

export function getSnapshot(): SystemSnapshot {
  return snapshot;
}

// --- Fetching ---

async function doRefresh(): Promise<void> {
  try {
    const [servicesResp, devices] = await Promise.all([
      api.listServices(),
      api.listDevices(),
    ]);
    setSnapshot({
      services: servicesResp.services,
      devices,
      status: "ready",
      error: null,
      hasData: true,
    });
  } catch (e) {
    setSnapshot({
      ...snapshot,
      status: "error",
      error: e instanceof Error ? e : new Error(String(e)),
    });
  }
}

/// Fetch services + devices and update the store. Deduplicated —
/// concurrent calls share the same in-flight promise.
export function refresh(): Promise<void> {
  if (fetchInFlight !== null) return fetchInFlight;
  fetchInFlight = doRefresh().finally(() => {
    fetchInFlight = null;
  });
  return fetchInFlight;
}

/// Start the first fetch if it hasn't been kicked off yet.
export function ensureInitialized(): void {
  if (initStarted) return;
  initStarted = true;
  void refresh();
}

// --- Polling fallback (when events are disconnected) ---

export function startPolling(intervalMs: number = 5_000): void {
  if (pollTimer !== null) return;
  pollTimer = setInterval(() => void refresh(), intervalMs);
}

export function stopPolling(): void {
  if (pollTimer === null) return;
  clearInterval(pollTimer);
  pollTimer = null;
}

// --- Event handling ---

/// Patch the store from a system event. Called by useEventsConnection
/// on every WebSocket message.
export function handleEvent(event: SystemEvent): void {
  switch (event.type) {
    case "state_changed": {
      const { service, to } = event;
      if (service && typeof to === "string") {
        patchServiceState(service, to);
      }
      scheduleRefresh();
      break;
    }
    case "allocation_changed": {
      const { service, reservations } = event;
      if (service && reservations && typeof reservations === "object") {
        patchReservations(service, reservations as Record<string, number>);
      }
      scheduleRefresh();
      break;
    }
    case "config_reloaded":
      // Services may have been added, removed, or structurally changed.
      // Immediate, not debounced.
      void refresh();
      break;
    // estimator_drift and overflow don't affect the services/devices list.
  }
}

// --- Internal helpers ---

function scheduleRefresh(debounceMs: number = 500): void {
  if (debounceTimer !== null) return;
  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    void refresh();
  }, debounceMs);
}

function patchServiceState(name: string, newState: string): void {
  const idx = snapshot.services.findIndex((s) => s.name === name);
  if (idx === -1) return;
  const services = snapshot.services.slice();
  services[idx] = { ...services[idx], state: newState };
  setSnapshot({ ...snapshot, services });
}

function patchReservations(
  service: string,
  reservations: Record<string, number>,
): void {
  const devices = snapshot.devices.map((d) => {
    const bytes = reservations[d.id];
    const existing = d.reservations.find((r) => r.service === service);
    if (bytes === undefined) {
      return {
        ...d,
        reservations: d.reservations.filter((r) => r.service !== service),
      };
    }
    const elastic = existing?.elastic ?? false;
    if (existing) {
      return {
        ...d,
        reservations: d.reservations.map((r) =>
          r.service === service ? { ...r, bytes, elastic } : r,
        ),
      };
    }
    return {
      ...d,
      reservations: [...d.reservations, { bytes, elastic, service }],
    };
  });
  setSnapshot({ ...snapshot, devices });
}

// --- Hooks ---

export function useServices(): StoreResult<ServiceSummary[]> {
  const snap = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  useEffect(() => {
    ensureInitialized();
  }, []);
  return {
    data: snap.hasData ? snap.services : undefined,
    isPending: snap.status === "loading",
    error: snap.error,
  };
}

export function useDevices(): StoreResult<DeviceSummary[]> {
  const snap = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  useEffect(() => {
    ensureInitialized();
  }, []);
  return {
    data: snap.hasData ? snap.devices : undefined,
    isPending: snap.status === "loading",
    error: snap.error,
  };
}
