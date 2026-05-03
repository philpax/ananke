// TanStack Query wrappers over `api` in `./client`. Picks sensible refetch
// cadences for a dashboard that's expected to be open while a human pokes
// services on and off — fast enough that state transitions feel
// instantaneous, slow enough that `/api/services` isn't being hit every
// frame.

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
  type LogsResponse,
  type ServiceDetail,
  type ServiceSummary,
  type ServicesResponse,
  type StartResponse,
  type StopResponse,
} from "./client.ts";

const SERVICES_POLL_MS = 2_000;
const DEVICES_POLL_MS = 2_000;
const LOGS_POLL_MS = 5_000;

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

export function useLogs(
  name: string | null,
): UseQueryResult<LogsResponse, Error> {
  return useQuery({
    queryKey: ["logs", name],
    queryFn: () => api.getLogs(name ?? ""),
    enabled: name !== null,
    refetchInterval: LOGS_POLL_MS,
  });
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
