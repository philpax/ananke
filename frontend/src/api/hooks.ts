// TanStack Query hooks over the management API. Services, devices, the
// event history, daemon info, and service detail are live queries kept
// current by the `/api/live` socket (see `live/`); logs follow over the same
// socket (see `logs/`). Everything else uses TanStack Query with sensible
// refetch cadences.

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
  type DeviceSummary,
  type EventHistory,
  type ServiceRestartEntry,
  type DisableResponse,
  type EnableResponse,
  type LaunchCommandResponse,
  type MetricsQuery,
  type MetricsResponse,
  type ServiceDetail,
  type ServiceSummary,
  type StartResponse,
  type StopResponse,
  type ValidationError,
} from "./client.ts";
import {
  devicesQuery,
  eventsQuery,
  infoQuery,
  serviceDetailQuery,
  servicesQuery,
} from "./live/resources.ts";
import { useLive } from "./live/useLive.ts";

const METRICS_STALE_MS = 30_000;

export function useServices(): UseQueryResult<ServiceSummary[], Error> {
  return useLive(servicesQuery, { select: selectServices });
}

function selectServices(response: { services: ServiceSummary[] }) {
  return response.services;
}

export function useDevices(): UseQueryResult<DeviceSummary[], Error> {
  return useLive(devicesQuery);
}

export function useEventFeed(): UseQueryResult<EventHistory, Error> {
  return useLive(eventsQuery);
}

export function useInfo(): UseQueryResult<DaemonInfoResponse, Error> {
  return useLive(infoQuery);
}

export function useServiceDetail(
  name: string | null,
): UseQueryResult<ServiceDetail, Error> {
  return useLive(serviceDetailQuery(name ?? ""), { enabled: name !== null });
}

export function useServiceCommand(
  name: string | null,
  enabled: boolean,
): UseQueryResult<LaunchCommandResponse, Error> {
  return useQuery({
    queryKey: ["service-command", name],
    queryFn: () => api.serviceCommand(name ?? ""),
    enabled: enabled && name !== null,
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

export function useRestarts(
  service?: string,
  since?: number,
  until?: number,
): UseQueryResult<ServiceRestartEntry[], Error> {
  return useQuery({
    queryKey: ["restarts", service, since, until],
    queryFn: () =>
      api.getRestarts(service, since, until).then((r) => r.restarts),
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

export type LifecycleAction =
  | "start"
  | "stop"
  | "restart"
  | "enable"
  | "disable";

export type LifecycleResponse =
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
        default: {
          const _: never = action;
          return _;
        }
      }
    },
    // The live socket pushes the outcome; this covers the socket being down.
    onSettled: (_data, _err, vars) => {
      void qc.invalidateQueries({ queryKey: ["service-detail", vars.name] });
      void qc.invalidateQueries({ queryKey: ["services"] });
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
