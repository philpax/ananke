// React adapter for the live socket. A live query is an ordinary TanStack
// query whose cache entry the socket overwrites on every snapshot. HTTP is
// the first load and the fallback: while the socket is down the query polls,
// and once it is open polling stops and snapshots take over.

import {
  createContext,
  useContext,
  useEffect,
  useSyncExternalStore,
} from "react";
import {
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";

import type { LiveQueryDef } from "./resources.ts";
import { liveSocket, type LiveSocket, type LiveStatus } from "./socket.ts";

const FALLBACK_POLL_MS = 5_000;

/// The connection live queries use. Tests provide their own.
export const LiveSocketContext = createContext<LiveSocket>(liveSocket);

export function useLiveSocket(): LiveSocket {
  return useContext(LiveSocketContext);
}

export function useLiveStatus(): LiveStatus {
  const socket = useLiveSocket();
  return useSyncExternalStore(
    socket.onChange,
    socket.getStatus,
    socket.getStatus,
  );
}

export function useDaemonUpdated(): boolean {
  const socket = useLiveSocket();
  return useSyncExternalStore(
    socket.onChange,
    socket.getDaemonUpdated,
    socket.getDaemonUpdated,
  );
}

/// The daemon build this tab first connected to, once its hello arrives.
export function useDaemonBuild(): string | null {
  const socket = useLiveSocket();
  return useSyncExternalStore(
    socket.onChange,
    socket.getBuild,
    socket.getBuild,
  );
}

export type UseLiveOptions<T, S> = {
  readonly select?: (data: T) => S;
  readonly enabled?: boolean;
};

export function useLive<T, S = T>(
  def: LiveQueryDef<T>,
  options: UseLiveOptions<T, S> = {},
): UseQueryResult<S, Error> {
  const socket = useLiveSocket();
  const open = useLiveStatus() === "open";
  const qc = useQueryClient();
  const enabled = options.enabled ?? true;

  const query = useQuery<T, Error, S>({
    queryKey: def.queryKey,
    queryFn: def.fetch,
    enabled,
    staleTime: open ? Infinity : 0,
    refetchInterval: open ? false : FALLBACK_POLL_MS,
    select: options.select,
  });

  useEffect(() => {
    if (!enabled) return;
    return socket.subscribe(def.resource, {
      onSnapshot: (data) => {
        const body = def.extract(data);
        if (body !== undefined) qc.setQueryData(def.queryKey, body);
      },
      // Refetch over HTTP so the query carries the same error a reload
      // would show; the next snapshot replaces it.
      onError: () => {
        void qc.invalidateQueries({ queryKey: def.queryKey, exact: true });
      },
    });
  }, [socket, def, enabled, qc]);

  return query;
}
