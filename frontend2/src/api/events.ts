// System-wide event subscription via the `/api/events` WebSocket.
// The daemon publishes state changes, allocation shifts, config
// reloads, and estimator drift events. This hook opens a single
// connection on mount and drives TanStack Query invalidation so
// the UI updates near-instantly without polling.
//
// The connection state is tracked in a module-level flag so that
// query hooks can disable polling while the WebSocket is live.

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "./client.ts";

export type SystemEvent = {
  type: string;
  service?: string;
  timestamp: number;
  [key: string]: unknown;
};

export type EventsState = {
  events: SystemEvent[];
  connected: boolean;
};

const MAX_EVENTS = 1_000;

let eventsConnected = false;

export function isEventsConnected(): boolean {
  return eventsConnected;
}

export function useEvents(): EventsState {
  const [state, setState] = useState<EventsState>({
    events: [],
    connected: false,
  });
  const qc = useQueryClient();
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      socket = new WebSocket(api.eventsStreamUrl());
      socket.onopen = () => {
        eventsConnected = true;
        setState((prev) => ({ ...prev, connected: true }));
        // Invalidate to trigger a refetch; the refetchInterval
        // function will now return false (WS connected), stopping
        // polling until the socket drops again.
        void qc.invalidateQueries({ queryKey: ["services"] });
        void qc.invalidateQueries({ queryKey: ["devices"] });
      };
      socket.onclose = () => {
        if (cancelled.current) return;
        eventsConnected = false;
        setState((prev) => ({ ...prev, connected: false }));
        // Invalidate to trigger a refetch; the refetchInterval
        // function will now return the fallback poll interval,
        // restarting polling until the socket reconnects.
        void qc.invalidateQueries({ queryKey: ["services"] });
        void qc.invalidateQueries({ queryKey: ["devices"] });
        reconnectTimer = window.setTimeout(connect, 2_000);
      };
      socket.onerror = () => {
        setState((prev) => ({ ...prev, connected: false }));
      };
      socket.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        let event: SystemEvent;
        try {
          event = JSON.parse(ev.data) as SystemEvent;
        } catch {
          return;
        }
        setState((prev) => {
          const events = [...prev.events, event];
          if (events.length > MAX_EVENTS) {
            events.splice(0, events.length - MAX_EVENTS);
          }
          return { ...prev, events };
        });

        // Drive query invalidation based on event type.
        switch (event.type) {
          case "state_changed":
          case "allocation_changed":
            void qc.invalidateQueries({ queryKey: ["services"] });
            void qc.invalidateQueries({ queryKey: ["devices"] });
            if (event.service) {
              void qc.invalidateQueries({
                queryKey: ["service-detail", event.service],
              });
            }
            break;
          case "config_reloaded":
            void qc.invalidateQueries({ queryKey: ["config"] });
            void qc.invalidateQueries({ queryKey: ["services"] });
            void qc.invalidateQueries({ queryKey: ["devices"] });
            break;
          case "estimator_drift":
            if (event.service) {
              void qc.invalidateQueries({
                queryKey: ["service-detail", event.service],
              });
            }
            break;
        }
      };
    };
    connect();

    return () => {
      cancelled.current = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [qc]);

  return state;
}
