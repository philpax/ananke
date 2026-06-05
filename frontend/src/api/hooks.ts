// TanStack Query wrappers over `api` in `./client`. Picks sensible refetch
// cadences for a dashboard that's expected to be open while a human pokes
// services on and off — fast enough that state transitions feel
// instantaneous, slow enough that `/api/services` isn't being hit every
// frame.

import { useEffect, useMemo, useRef, useState } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from "@tanstack/react-query";

import {
  api,
  type ConfigResponse,
  type DeviceSummary,
  type DisableResponse,
  type EnableResponse,
  type LaunchCommand,
  type LogLine,
  type LogStreamMessage,
  type ServiceDetail,
  type ServiceSummary,
  type ServicesResponse,
  type StartResponse,
  type StopResponse,
} from "./client.ts";

const SERVICES_POLL_MS = 2_000;
const DEVICES_POLL_MS = 2_000;
/// Page size when fetching the historical seed for a logs window.
/// Matches the daemon's MAX_LIMIT so a "last 1 h on a chatty service"
/// view typically fits in one page.
const LOGS_PAGE_SIZE = 1_000;
/// Cap on how many seed pages we'll fetch back from `until`. Bounds the
/// initial cost of opening a huge window on a long-running service —
/// past this point the user gets the most-recent N × page-size lines
/// and a note that older lines are out of view.
const LOGS_MAX_SEED_PAGES = 5;

export function useServices(): UseQueryResult<ServiceSummary[], Error> {
  return useQuery({
    queryKey: ["services"],
    queryFn: () =>
      api.listServices().then((resp: ServicesResponse) => resp.services),
    refetchInterval: SERVICES_POLL_MS,
  });
}

export function useDevices(): UseQueryResult<DeviceSummary[], Error> {
  return useQuery({
    queryKey: ["devices"],
    queryFn: api.listDevices,
    refetchInterval: DEVICES_POLL_MS,
  });
}

export function useServiceDetail(
  name: string | null,
): UseQueryResult<ServiceDetail, Error> {
  return useQuery({
    queryKey: ["service-detail", name],
    queryFn: () => api.serviceDetail(name ?? ""),
    enabled: name !== null,
    refetchInterval: SERVICES_POLL_MS,
  });
}

/// Fetch the launch command ananke uses (or would use) for a service.
/// Gated by `enabled` so the daemon only runs the estimator/packer when the
/// user actually reveals the command, and re-polls while open so the
/// `running` ↔ `preview` source and any placement shift stay current.
export function useServiceCommand(
  name: string | null,
  enabled: boolean,
): UseQueryResult<LaunchCommand, Error> {
  return useQuery({
    queryKey: ["service-command", name],
    queryFn: () => api.serviceCommand(name ?? ""),
    enabled: enabled && name !== null,
    refetchInterval: enabled ? SERVICES_POLL_MS : false,
  });
}

/// A time window over the service log stream. The seed REST query
/// covers `[since, until ?? now]`; when `until` is absent the window
/// is "live" and the hook subscribes to the WebSocket for new lines.
export type LogWindow =
  | { kind: "relative"; durationMs: number }
  | { kind: "absolute"; sinceMs: number; untilMs: number | null };

export type LogWindowResult = {
  /// Lines in chronological order (oldest first, newest last).
  lines: readonly LogLine[];
  /// `true` when the WebSocket is actively appending new lines.
  isLive: boolean;
  /// Lines the WebSocket dropped because the subscriber lagged.
  /// Cumulative since the hook subscribed; reset on window change.
  droppedByOverflow: number;
  /// `true` when the seed REST page reported `next_cursor`, i.e. the
  /// window contains older lines past the seed page cap.
  truncated: boolean;
  /// Daemon-side fetch error, if any. WebSocket errors are surfaced
  /// via `isLive: false` rather than thrown.
  error: Error | null;
  loading: boolean;
};

/// Fetch a seeded + live-tailed log window for a service. Combines:
///   * a paginated REST seed of `[since, until]` (newest-first from
///     the daemon, reversed to chronological client-side),
///   * a WebSocket subscription that appends new lines while the
///     window is live (i.e. `untilMs == null`).
///
/// Lines are deduplicated by `(run_id, seq)` — the seed page may have
/// fetched lines that the WS then re-delivers, and we keep one copy.
export function useLogWindow(
  name: string | null,
  window: LogWindow,
): LogWindowResult {
  // Resolve the window to absolute bounds *and* to a stable cache key.
  // `relative` is always re-anchored to `now` at subscription time so
  // the seed page covers exactly `[now - duration, now]`; live WS
  // lines after that point keep extending the window.
  const bounds = useMemo(() => resolveBounds(window), [window]);

  // REST seed.
  const seed = useQuery({
    queryKey: ["log-seed", name, bounds.sinceMs, bounds.untilMs],
    queryFn: () =>
      fetchSeedWindow(name ?? "", bounds.sinceMs, bounds.untilMs ?? Date.now()),
    enabled: name !== null,
  });

  // WS live tail when the window is open-ended.
  const isLiveWindow = bounds.untilMs === null;
  const ws = useLogStream(isLiveWindow ? name : null);

  // Merge + dedupe. Lines outside the window are filtered out so a
  // 5-minute relative window doesn't accumulate WS lines forever.
  const lines = useMemo(() => {
    const map = new Map<string, LogLine>();
    const lowerBound = bounds.sinceMs;
    const upperBound = bounds.untilMs;
    const accept = (l: LogLine) => {
      if (l.timestamp_ms < lowerBound) return;
      if (upperBound !== null && l.timestamp_ms > upperBound) return;
      map.set(`${l.run_id}:${l.seq}`, l);
    };
    for (const l of seed.data?.lines ?? []) accept(l);
    for (const l of ws.lines) accept(l);
    return Array.from(map.values()).sort(compareLogLines);
  }, [seed.data, ws.lines, bounds.sinceMs, bounds.untilMs]);

  return {
    lines,
    isLive: isLiveWindow && ws.connected,
    droppedByOverflow: ws.dropped,
    truncated: seed.data?.truncated ?? false,
    error: seed.error,
    loading: seed.isPending && name !== null,
  };
}

type SeedWindow = { lines: LogLine[]; truncated: boolean };

async function fetchSeedWindow(
  name: string,
  sinceMs: number,
  untilMs: number,
): Promise<SeedWindow> {
  // Page backward from `until` using the daemon's `before` cursor until
  // either the cursor is exhausted (we reached `since`) or we hit the
  // page cap. The daemon returns newest-first; we accumulate raw and
  // let the merge stage sort to chronological.
  const collected: LogLine[] = [];
  let cursor: string | undefined;
  let truncated = false;
  for (let page = 0; page < LOGS_MAX_SEED_PAGES; page++) {
    const resp = await api.getLogs(name, {
      since: sinceMs,
      until: untilMs,
      limit: LOGS_PAGE_SIZE,
      before: cursor,
    });
    collected.push(...resp.logs);
    if (!resp.next_cursor) {
      cursor = undefined;
      break;
    }
    cursor = resp.next_cursor;
  }
  if (cursor) {
    // We bailed at the page cap with the daemon reporting more lines
    // behind us — flag for the UI so it can render a "truncated" hint.
    truncated = true;
  }
  return { lines: collected, truncated };
}

type LogStreamState = {
  lines: LogLine[];
  dropped: number;
  connected: boolean;
};

/// Subscribe to `/logs/stream`. Returns the buffer of received lines
/// plus the dropped-frame counter from `LogStreamMessage::overflow`.
/// Reconnects on close with a 1 s backoff until the consumer
/// switches away (passes `null` for `name`). Buffer is reset every
/// time the subscription target changes.
function useLogStream(name: string | null): LogStreamState {
  // The buffer is per *hook instance*: `LogsView` is keyed by service
  // (via `ServiceDetailInline`'s parent), so switching services
  // remounts this hook and gives us a fresh subscription. Within one
  // instance `name` only toggles between `null` (live mode off) and
  // the fixed service name, so we don't need to clear the buffer on
  // change — any lines that arrived during a previous live segment
  // either still match the current window (and we'd want to see them
  // again) or fall out via the timestamp filter in `useLogWindow`.
  const [state, setState] = useState<LogStreamState>(EMPTY_STREAM_STATE);
  const cancelled = useRef(false);

  useEffect(() => {
    if (name === null) {
      // No subscription target — exit without touching state.
      return;
    }
    cancelled.current = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      socket = new WebSocket(api.logStreamUrl(name));
      socket.onopen = () => setState((prev) => ({ ...prev, connected: true }));
      socket.onclose = () => {
        if (cancelled.current) return;
        setState((prev) => ({ ...prev, connected: false }));
        // Cheap fixed backoff — the WS endpoint runs on the same daemon
        // so reconnect-loop-on-bad-state isn't a concern.
        reconnectTimer = window.setTimeout(connect, 1_000);
      };
      socket.onerror = () => {
        // Mirror onclose; the browser fires close right after.
        setState((prev) => ({ ...prev, connected: false }));
      };
      socket.onmessage = (ev) => {
        const data = ev.data;
        if (typeof data !== "string") return;
        let parsed: LogStreamMessage;
        try {
          parsed = JSON.parse(data) as LogStreamMessage;
        } catch {
          return;
        }
        setState((prev) => {
          if (parsed.type === "overflow") {
            return { ...prev, dropped: prev.dropped + parsed.dropped };
          }
          // `type === "line"` — the flattened shape is `LogLine & { type }`.
          const line: LogLine = {
            timestamp_ms: parsed.timestamp_ms,
            stream: parsed.stream,
            line: parsed.line,
            run_id: parsed.run_id,
            seq: parsed.seq,
          };
          return { ...prev, lines: [...prev.lines, line] };
        });
      };
    };
    connect();

    return () => {
      cancelled.current = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [name]);

  return state;
}

const EMPTY_STREAM_STATE: LogStreamState = {
  lines: [],
  dropped: 0,
  connected: false,
};

function resolveBounds(window: LogWindow): {
  sinceMs: number;
  untilMs: number | null;
} {
  if (window.kind === "relative") {
    const now = Date.now();
    return { sinceMs: now - window.durationMs, untilMs: null };
  }
  return { sinceMs: window.sinceMs, untilMs: window.untilMs };
}

function compareLogLines(a: LogLine, b: LogLine): number {
  if (a.timestamp_ms !== b.timestamp_ms) return a.timestamp_ms - b.timestamp_ms;
  if (a.run_id !== b.run_id) return a.run_id - b.run_id;
  return a.seq - b.seq;
}

export function useConfig(): UseQueryResult<ConfigResponse, Error> {
  return useQuery({
    queryKey: ["config"],
    queryFn: api.getConfig,
  });
}

type LifecycleAction = "start" | "stop" | "restart" | "enable" | "disable";

type LifecycleResponse =
  | StartResponse
  | StopResponse
  | EnableResponse
  | DisableResponse;

export function useLifecycle(): UseMutationResult<
  LifecycleResponse,
  Error,
  { action: LifecycleAction; name: string }
> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({
      action,
      name,
    }: {
      action: LifecycleAction;
      name: string;
    }): Promise<LifecycleResponse> => {
      switch (action) {
        case "start":
          return api.start(name);
        case "stop":
          return api.stop(name);
        case "restart":
          return api.restart(name);
        case "enable":
          return api.enable(name);
        case "disable":
          return api.disable(name);
      }
    },
    // Refetch everything that could have shifted after a lifecycle poke.
    // Devices because reservations changed; services + detail because
    // state transitions are the whole point.
    onSettled: (_data, _err, vars) => {
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["devices"] });
      void qc.invalidateQueries({ queryKey: ["service-detail", vars.name] });
    },
  });
}
