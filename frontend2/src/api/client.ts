// Typed fetch helpers over the management API. Request + response shapes
// come from `src/api/types.ts`, which is regenerated from the daemon's
// OpenAPI document via `npm run gen-types`.

import type { components, paths } from "./types.ts";

type Schemas = components["schemas"];

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
export type ConfigValidateResponse = Schemas["ConfigValidateResponse"];
export type ValidationError = Schemas["ValidationError"];
export type ApiError = Schemas["ApiError"];
export type MetricsResponse = Schemas["MetricsResponse"];
export type MetricBucketResponse = Schemas["MetricBucketResponse"];
export type DeviceSamplesResponse = Schemas["DeviceSamplesResponse"];
export type DeviceSampleResponse = Schemas["DeviceSampleResponse"];
export type DaemonInfoResponse = Schemas["DaemonInfoResponse"];
export type OneshotRequest = Schemas["OneshotRequest"];
export type OneshotResponse = Schemas["OneshotResponse"];
export type OneshotStatus = Schemas["OneshotStatus"];

export type LogsQuery = {
  since?: number;
  until?: number;
  run?: number;
  stream?: "stdout" | "stderr";
  limit?: number;
  before?: string;
};

export type MetricsQuery = {
  service?: string;
  since?: number;
  until?: number;
  bucket?: string;
};

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(path, { headers: { accept: "application/json" } });
  if (!resp.ok) throw new Error(await errorMessage(resp));
  return (await resp.json()) as T;
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const resp = await fetch(path, {
    method: "POST",
    headers: {
      accept: "application/json",
      ...(body !== undefined ? { "content-type": "application/json" } : {}),
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  if (!resp.ok && resp.status !== 202)
    throw new Error(await errorMessage(resp));
  return (await resp.json()) as T;
}

async function errorMessage(resp: Response): Promise<string> {
  const fallback = `${resp.status} ${resp.statusText}`;
  try {
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

export type ConfigSaveResult =
  | { kind: "ok" }
  | { kind: "hash_mismatch"; serverHash: string }
  | { kind: "validation_errors"; errors: ValidationError[] }
  | { kind: "error"; message: string };

async function putConfigRaw(
  toml: string,
  hash: string,
): Promise<ConfigSaveResult> {
  const resp = await fetch("/api/config", {
    method: "PUT",
    headers: {
      "content-type": "text/plain",
      "if-match": `"${hash}"`,
    },
    body: toml,
  });
  if (resp.status === 202) return { kind: "ok" };
  if (resp.status === 412) {
    const etag = resp.headers.get("etag");
    const serverHash = etag ? etag.replace(/"/g, "") : "";
    return { kind: "hash_mismatch", serverHash };
  }
  if (resp.status === 422) {
    const body = (await resp.json()) as ConfigValidateResponse;
    return { kind: "validation_errors", errors: body.errors };
  }
  return { kind: "error", message: await errorMessage(resp) };
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
  logStreamUrl: (name: string) => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/api/services/${encodeURIComponent(name)}/logs/stream`;
  },
  getConfig: () => getJson<ConfigResponse>("/api/config"),
  validateConfig: (content: string) =>
    postJson<ConfigValidateResponse>("/api/config/validate", { content }),
  putConfig: (toml: string, hash: string) => putConfigRaw(toml, hash),
  getMetrics: (query: MetricsQuery = {}) => {
    const params = new URLSearchParams();
    if (query.service !== undefined) params.set("service", query.service);
    if (query.since !== undefined) params.set("since", String(query.since));
    if (query.until !== undefined) params.set("until", String(query.until));
    if (query.bucket !== undefined) params.set("bucket", query.bucket);
    const qs = params.toString();
    return getJson<MetricsResponse>(`/api/metrics${qs ? `?${qs}` : ""}`);
  },
  getDeviceSamples: (device?: string, since?: number, until?: number) => {
    const params = new URLSearchParams();
    if (device !== undefined) params.set("device", device);
    if (since !== undefined) params.set("since", String(since));
    if (until !== undefined) params.set("until", String(until));
    const qs = params.toString();
    return getJson<DeviceSamplesResponse>(
      `/api/devices/samples${qs ? `?${qs}` : ""}`,
    );
  },
  getPrometheusMetrics: () => fetch("/metrics").then((r) => r.text()),
  getInfo: () => getJson<DaemonInfoResponse>("/api/info"),
  listOneshots: () =>
    getJson<
      paths["/api/oneshot"]["get"]["responses"]["200"]["content"]["application/json"]
    >("/api/oneshot"),
  createOneshot: (req: OneshotRequest) =>
    postJson<OneshotResponse>("/api/oneshot", req),
  getOneshot: (id: string) =>
    getJson<OneshotStatus>(`/api/oneshot/${encodeURIComponent(id)}`),
  deleteOneshot: async (id: string): Promise<void> => {
    const resp = await fetch(`/api/oneshot/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!resp.ok && resp.status !== 204) {
      throw new Error(await errorMessage(resp));
    }
  },
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
  eventsStreamUrl: () => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/api/events`;
  },
};
