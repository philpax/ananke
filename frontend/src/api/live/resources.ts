// One definition per live resource: the TanStack query key it fills, the
// HTTP fetch used for the first load and while the socket is down, and how to
// pick its body out of a snapshot. Query keys keep the names the rest of the
// app already invalidates.

import { api } from "../client.ts";
import type {
  DaemonInfoResponse,
  DeviceSummary,
  EventHistory,
  LiveData,
  LiveResource,
  ReservationsResponse,
  ServiceDetail,
  ServicesResponse,
} from "../client.ts";

export type LiveQueryDef<T> = {
  readonly queryKey: readonly unknown[];
  readonly resource: LiveResource;
  readonly fetch: () => Promise<T>;
  readonly extract: (data: LiveData) => T | undefined;
};

export const servicesQuery: LiveQueryDef<ServicesResponse> = {
  queryKey: ["services"],
  resource: { kind: "services" },
  fetch: api.listServices,
  extract: (data) => (data.resource === "services" ? data.body : undefined),
};

export const devicesQuery: LiveQueryDef<DeviceSummary[]> = {
  queryKey: ["devices"],
  resource: { kind: "devices" },
  fetch: api.listDevices,
  extract: (data) => (data.resource === "devices" ? data.body : undefined),
};

export const eventsQuery: LiveQueryDef<EventHistory> = {
  queryKey: ["events"],
  resource: { kind: "events" },
  fetch: api.getEventHistory,
  extract: (data) => (data.resource === "events" ? data.body : undefined),
};

export const infoQuery: LiveQueryDef<DaemonInfoResponse> = {
  queryKey: ["info"],
  resource: { kind: "info" },
  fetch: api.getInfo,
  extract: (data) => (data.resource === "info" ? data.body : undefined),
};

export const reservationsQuery: LiveQueryDef<ReservationsResponse> = {
  queryKey: ["reservations"],
  resource: { kind: "reservations" },
  fetch: api.listReservations,
  extract: (data) => (data.resource === "reservations" ? data.body : undefined),
};

// Cached per name so a definition keeps its identity across renders, which
// keeps `useLive`'s subscription effect from re-running needlessly.
const serviceDetailQueries = new Map<string, LiveQueryDef<ServiceDetail>>();

export function serviceDetailQuery(name: string): LiveQueryDef<ServiceDetail> {
  const cached = serviceDetailQueries.get(name);
  if (cached) return cached;
  const def: LiveQueryDef<ServiceDetail> = {
    queryKey: ["service-detail", name],
    resource: { kind: "service", name },
    fetch: () => api.serviceDetail(name),
    extract: (data) =>
      data.resource === "service" && data.body.name === name
        ? data.body
        : undefined,
  };
  serviceDetailQueries.set(name, def);
  return def;
}
