// TanStack Query hooks over the management API. Services and devices
// are served from the systemStore (event-driven, no polling when the
// WebSocket is connected). Everything else uses TanStack Query with
// sensible refetch cadences.

import { useMemo, useState } from "react";
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
  type ConfigSaveResult,
  type DaemonInfoResponse,
  type DeviceSampleResponse,
  type DisableResponse,
  type EnableResponse,
  type LaunchCommand,
  type LogLine,
  type LogStreamMessage,
  type MetricsQuery,
  type MetricsResponse,
  type OneshotRequest,
  type OneshotResponse,
  type OneshotStatus,
  type ServiceDetail,
  type StartResponse,
  type StopResponse,
  type ValidationError,
} from "./client.ts";
import { isEventsConnected } from "./events.ts";
import { refresh as refreshStore } from "./systemStore.ts";
import { useWebSocket } from "./useWebSocket.ts";

// Re-export so consumers keep importing from hooks.ts.
export { useServices, useDevices } from "./systemStore.ts";

const SERVICES_POLL_MS = 5_000;
const METRICS_STALE_MS = 30_000;
const LOGS_PAGE_SIZE = 1_000;
const LOGS_MAX_SEED_PAGES = 5;

export function useServiceDetail(
  name: string | null,
): UseQueryResult<ServiceDetail, Error> {
  return useQuery({
    queryKey: ["service-detail", name],
    queryFn: () => api.serviceDetail(name ?? ""),
    enabled: name !== null,
    refetchInterval: () => (isEventsConnected() ? false : SERVICES_POLL_MS),
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
    refetchInterval: enabled
      ? () => (isEventsConnected() ? false : SERVICES_POLL_MS)
      : false,
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
    placeholderData: (prev) => prev,
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
    placeholderData: (prev) => prev,
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
};

function useLogStream(name: string | null): LogStreamState & {
  connected: boolean;
} {
  const [state, setState] = useState<LogStreamState>(EMPTY_STREAM_STATE);

  const url = name !== null ? api.logStreamUrl(name) : null;
  const { connected } = useWebSocket(
    url,
    {
      onMessage: (data) => {
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
      },
    },
    1_000,
  );

  return { ...state, connected };
}

const EMPTY_STREAM_STATE: LogStreamState = {
  lines: [],
  dropped: 0,
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
      void refreshStore();
      void qc.invalidateQueries({ queryKey: ["service-detail", vars.name] });
    },
  });
}

export function useValidateConfig(): UseMutationResult<
  { valid: boolean; errors: ValidationError[] },
  Error,
  string
> {
  return useMutation({
    mutationFn: (content: string) => api.validateConfig(content),
  });
}

export function useSaveConfig(): UseMutationResult<
  ConfigSaveResult,
  Error,
  { content: string; hash: string }
> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ content, hash }) => api.putConfig(content, hash),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ["config"] });
    },
  });
}

const ONESHOT_POLL_MS = 5_000;

export function useOneshots(): UseQueryResult<OneshotStatus[], Error> {
  return useQuery({
    queryKey: ["oneshots"],
    queryFn: () => api.listOneshots(),
    refetchInterval: () => (isEventsConnected() ? false : ONESHOT_POLL_MS),
  });
}

export function useCreateOneshot(): UseMutationResult<
  OneshotResponse,
  Error,
  OneshotRequest
> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: OneshotRequest) => api.createOneshot(req),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ["oneshots"] });
    },
  });
}

export function useDeleteOneshot(): UseMutationResult<void, Error, string> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteOneshot(id),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ["oneshots"] });
    },
  });
}
