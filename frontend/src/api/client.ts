// Typed fetch helpers over the management API. Request + response shapes
// come from `src/api/types.ts`, which is regenerated from the daemon's
// OpenAPI document via `npm run gen-types`.

import type { components, paths } from "./types.ts";

type Schemas = components["schemas"];

export type ServiceSummary = Schemas["ServiceSummary"];
export type ServicesResponse = Schemas["ServicesResponse"];
export type ServiceDetail = Schemas["ServiceDetail"];
export type LogLine = Schemas["LogLine"];
export type LogsResponse = Schemas["LogsResponse"];
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
  try {
    const body = (await resp.json()) as ApiError;
    return body.error?.message ?? `${resp.status} ${resp.statusText}`;
  } catch {
    return `${resp.status} ${resp.statusText}`;
  }
}

export const api = {
  listServices: () =>
    getJson<
      paths["/api/services"]["get"]["responses"]["200"]["content"]["application/json"]
    >("/api/services"),
  serviceDetail: (name: string) =>
    getJson<ServiceDetail>(`/api/services/${encodeURIComponent(name)}`),
  listDevices: () =>
    getJson<
      paths["/api/devices"]["get"]["responses"]["200"]["content"]["application/json"]
    >("/api/devices"),
  getLogs: (name: string) =>
    getJson<LogsResponse>(`/api/services/${encodeURIComponent(name)}/logs`),
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
