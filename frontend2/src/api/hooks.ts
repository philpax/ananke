// TanStack Query hooks over the management API. Picks sensible refetch
// cadences for a dashboard that's expected to be open while a human
// pokes services on and off — fast enough that state transitions feel
// instantaneous, slow enough that `/api/services` isn't being hit every
// frame. The events WebSocket drives invalidation for near-instant
// updates; polling is a fallback.

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
  type DaemonInfoResponse,
  type DeviceSampleResponse,
  type DeviceSummary,
  type DisableResponse,
  type EnableResponse,
  type LaunchCommand,
  type LogLine,
  type LogStreamMessage,
  type MetricsQuery,
  type MetricsResponse,
  type ServiceDetail,
  type ServiceSummary,
  type ServicesResponse,
  type StartResponse,
  type StopResponse,
} from "./client.ts";

const SERVICES_POLL_MS = 2_000;
const DEVICES_POLL_MS = 2_000;
const METRICS_STALE_MS = 30_000;
const LOGS_PAGE_SIZE = 1_000;
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

export function useMetrics(
  query: MetricsQuery = {},
): UseQueryResult<MetricsResponse, Error> {
  return useQuery({
    queryKey: [
      "metrics",
      query.service,
      query.since,
      query.until,
      query.bucket,
    ],
    queryFn: () => api.getMetrics(query),
    staleTime: METRICS_STALE_MS,
    refetchInterval: METRICS_STALE_MS,
  });
}

export function useDeviceSamples(
  device?: string,
  since?: number,
  until?: number,
): UseQueryResult<DeviceSampleResponse[], Error> {
  return useQuery({
    queryKey: ["device-samples", device, since, until],
    queryFn: () =>
      api.getDeviceSamples(device, since, until).then((r) => r.samples),
    staleTime: METRICS_STALE_MS,
    refetchInterval: METRICS_STALE_MS,
  });
}

export function useConfig(): UseQueryResult<ConfigResponse, Error> {
  return useQuery({
    queryKey: ["config"],
    queryFn: api.getConfig,
  });
}

export function useInfo(): UseQueryResult<DaemonInfoResponse, Error> {
  return useQuery({
    queryKey: ["info"],
    queryFn: api.getInfo,
    staleTime: Infinity,
  });
}

export type LogWindow =
  | { kind: "relative"; durationMs: number }
  | { kind: "absolute"; sinceMs: number; untilMs: number | null };

export type LogWindowResult = {
  lines: readonly LogLine[];
  isLive: boolean;
  droppedByOverflow: number;
  truncated: boolean;
  error: Error | null;
  loading: boolean;
};

export function useLogWindow(
  name: string | null,
  window: LogWindow,
): LogWindowResult {
  const bounds = useMemo(() => resolveBounds(window), [window]);

  const seed = useQuery({
    queryKey: ["log-seed", name, bounds.sinceMs, bounds.untilMs],
    queryFn: () =>
      fetchSeedWindow(name ?? "", bounds.sinceMs, bounds.untilMs ?? Date.now()),
    enabled: name !== null,
  });

  const isLiveWindow = bounds.untilMs === null;
  const ws = useLogStream(isLiveWindow ? name : null);

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
  if (cursor) truncated = true;
  return { lines: collected, truncated };
}

type LogStreamState = {
  lines: LogLine[];
  dropped: number;
  connected: boolean;
};

function useLogStream(name: string | null): LogStreamState {
  const [state, setState] = useState<LogStreamState>(EMPTY_STREAM_STATE);
  const cancelled = useRef(false);

  useEffect(() => {
    if (name === null) return;
    cancelled.current = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      socket = new WebSocket(api.logStreamUrl(name));
      socket.onopen = () => setState((prev) => ({ ...prev, connected: true }));
      socket.onclose = () => {
        if (cancelled.current) return;
        setState((prev) => ({ ...prev, connected: false }));
        reconnectTimer = window.setTimeout(connect, 1_000);
      };
      socket.onerror = () => {
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
    onSettled: (_data, _err, vars) => {
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["devices"] });
      void qc.invalidateQueries({ queryKey: ["service-detail", vars.name] });
    },
  });
}
