// Typed fetch helpers over the management API. Request + response shapes
// come from `src/api/types.ts`, which is regenerated from the daemon's
// OpenAPI document via `npm run gen-types`.

import type { components, paths } from "./types.ts";

type Schemas = components["schemas"];

/// Query parameters for `GET /api/services/{name}/logs`. Mirrors the
/// daemon's `LogsQuery` (see `ananke/src/api/management/logs.rs`).
/// Times are millisecond UNIX timestamps. `before` is an opaque cursor
/// from a previous response's `next_cursor` — the daemon pages
/// backward (older) from there.
export type LogsQuery = {
  since?: number;
  until?: number;
  run?: number;
  stream?: "stdout" | "stderr";
  limit?: number;
  before?: string;
};

export type ServiceSummary = Schemas["ServiceSummary"];
export type ServicesResponse = Schemas["ServicesResponse"];
export type ServiceDetail = Schemas["ServiceDetail"];
export type LaunchCommand = Schemas["LaunchCommand"];
export type ModelInfo = Schemas["ModelInfo"];
export type EstimateSummary = Schemas["EstimateSummary"];
export type PlacementPreview = Schemas["PlacementPreview"];
export type DevicePlacement = Schemas["DevicePlacement"];
export type LogLine = Schemas["LogLine"];
export type LogsResponse = Schemas["LogsResponse"];
export type LogStreamMessage = Schemas["LogStreamMessage"];
export type DeviceSummary = Schemas["DeviceSummary"];
export type DeviceReservation = Schemas["DeviceReservation"];
export type StartResponse = Schemas["StartResponse"];
export type StopResponse = Schemas["StopResponse"];
export type EnableResponse = Schemas["EnableResponse"];
export type DisableResponse = Schemas["DisableResponse"];
export type ConfigResponse = Schemas["ConfigResponse"];
export type ApiError = Schemas["ApiError"];

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(path, { headers: { accept: "application/json" } });
  if (!resp.ok) throw new Error(await errorMessage(resp));
  return (await resp.json()) as T;
}

async function postJson<T>(path: string): Promise<T> {
  const resp = await fetch(path, {
    method: "POST",
    headers: { accept: "application/json" },
  });
  if (!resp.ok && resp.status !== 202)
    throw new Error(await errorMessage(resp));
  return (await resp.json()) as T;
}

async function errorMessage(resp: Response): Promise<string> {
  const fallback = `${resp.status} ${resp.statusText}`;
  try {
    // Two error shapes are in play: the structured `ApiError`
    // (`{ error: { message } }`) and the simpler inline form some handlers
    // return (`{ error: "..." }`, e.g. the launch-command 422). Handle both.
    const body = (await resp.json()) as { error?: unknown };
    const e = body.error;
    if (typeof e === "string") return e;
    if (e && typeof e === "object" && "message" in e) {
      const message = (e as { message?: unknown }).message;
      if (typeof message === "string") return message;
    }
    return fallback;
  } catch {
    return fallback;
  }
}

export const api = {
  listServices: () =>
    getJson<
      paths["/api/services"]["get"]["responses"]["200"]["content"]["application/json"]
    >("/api/services"),
  serviceDetail: (name: string) =>
    getJson<ServiceDetail>(`/api/services/${encodeURIComponent(name)}`),
  serviceCommand: (name: string) =>
    getJson<LaunchCommand>(`/api/services/${encodeURIComponent(name)}/command`),
  listDevices: () =>
    getJson<
      paths["/api/devices"]["get"]["responses"]["200"]["content"]["application/json"]
    >("/api/devices"),
  getLogs: (name: string, query: LogsQuery = {}) => {
    const params = new URLSearchParams();
    if (query.since !== undefined) params.set("since", String(query.since));
    if (query.until !== undefined) params.set("until", String(query.until));
    if (query.run !== undefined) params.set("run", String(query.run));
    if (query.stream !== undefined) params.set("stream", query.stream);
    if (query.limit !== undefined) params.set("limit", String(query.limit));
    if (query.before !== undefined) params.set("before", query.before);
    const qs = params.toString();
    const path = `/api/services/${encodeURIComponent(name)}/logs${qs ? `?${qs}` : ""}`;
    return getJson<LogsResponse>(path);
  },
  /// Build the absolute WebSocket URL for `/logs/stream`. The frontend
  /// is served from the same host:port as the management API, so the
  /// scheme is mirrored (`ws` ↔ `http`, `wss` ↔ `https`).
  logStreamUrl: (name: string) => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/api/services/${encodeURIComponent(name)}/logs/stream`;
  },
  getConfig: () => getJson<ConfigResponse>("/api/config"),
  start: (name: string) =>
    postJson<StartResponse>(`/api/services/${encodeURIComponent(name)}/start`),
  stop: (name: string) =>
    postJson<StopResponse>(`/api/services/${encodeURIComponent(name)}/stop`),
  restart: (name: string) =>
    postJson<StartResponse>(
      `/api/services/${encodeURIComponent(name)}/restart`,
    ),
  enable: (name: string) =>
    postJson<EnableResponse>(
      `/api/services/${encodeURIComponent(name)}/enable`,
    ),
  disable: (name: string) =>
    postJson<DisableResponse>(
      `/api/services/${encodeURIComponent(name)}/disable`,
    ),
};
