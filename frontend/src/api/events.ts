// System-wide event subscription via the `/api/events` WebSocket.
// The daemon publishes state changes, allocation shifts, config
// reloads, and estimator drift events.
//
// Two exports:
//   - useEventsConnection(): opens the socket on app mount, feeds
//     events into the systemStore and drives TanStack Query
//     invalidation for non-store queries (config, service-detail).
//     Called once in App.
//   - useEventFeed(): subscribes to the in-memory event buffer for the
//     events tab. Does not open a second socket.

import { useEffect, useState, useSyncExternalStore } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { api } from "./client.ts";
import {
  handleEvent,
  refresh as refreshStore,
  startPolling,
  stopPolling,
} from "./systemStore.ts";
import { useWebSocket } from "./useWebSocket.ts";

export type SystemEvent = {
  type: string;
  service?: string;
  at_ms?: number;
  [key: string]: unknown;
};

const MAX_EVENTS = 1_000;

// --- Module-level event store ---

type Listener = () => void;

// Reassigned to a new array on each emit so useSyncExternalStore's
// Object.is check detects the change — mutating in place would keep
// the same reference and React would skip the re-render.
let eventBuffer: readonly SystemEvent[] = [];
const listeners = new Set<Listener>();
let eventsConnected = false;

function emit(event: SystemEvent) {
  const next =
    eventBuffer.length >= MAX_EVENTS
      ? eventBuffer.slice(eventBuffer.length - MAX_EVENTS + 1)
      : eventBuffer.slice();
  next.push(event);
  eventBuffer = next;
  for (const l of listeners) l();
}

function subscribe(l: Listener): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

function getSnapshot(): readonly SystemEvent[] {
  return eventBuffer;
}

export function isEventsConnected(): boolean {
  return eventsConnected;
}

// --- Force reconnect ---

let reconnectFn: (() => void) | null = null;

export function reconnectEvents(): void {
  reconnectFn?.();
}

// --- Hooks ---

/// Opens the events WebSocket on mount and drives query invalidation.
/// Call once in App. Does not return event data.
export function useEventsConnection(): void {
  const qc = useQueryClient();

  const { reconnect } = useWebSocket(
    api.eventsStreamUrl(),
    {
      onOpen: () => {
        eventsConnected = true;
        for (const l of listeners) l();
        stopPolling();
        void refreshStore();
        void qc.invalidateQueries({ queryKey: ["config"] });
        void qc.invalidateQueries({ queryKey: ["service-detail"] });
      },
      onClose: () => {
        eventsConnected = false;
        for (const l of listeners) l();
        startPolling();
      },
      onMessage: (data) => {
        let event: SystemEvent;
        try {
          event = JSON.parse(data) as SystemEvent;
        } catch {
          return;
        }
        emit(event);
        handleEvent(event);

        // Invalidate TanStack queries the store doesn't cover.
        switch (event.type) {
          case "state_changed":
          case "allocation_changed":
            if (event.service) {
              void qc.invalidateQueries({
                queryKey: ["service-detail", event.service],
              });
            }
            break;
          case "config_reloaded":
            void qc.invalidateQueries({ queryKey: ["config"] });
            break;
          case "estimator_drift":
            if (event.service) {
              void qc.invalidateQueries({
                queryKey: ["service-detail", event.service],
              });
            }
            break;
        }
      },
    },
    2_000,
  );

  useEffect(() => {
    reconnectFn = reconnect;
    return () => {
      reconnectFn = null;
    };
  }, [reconnect]);
}

/// Returns the current event buffer and connection state. Uses
/// useSyncExternalStore so multiple components can subscribe without
/// opening additional sockets.
export function useEventFeed(): {
  events: readonly SystemEvent[];
  connected: boolean;
} {
  const events = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const [connected, setConnected] = useState(eventsConnected);

  useEffect(() => {
    const l = () => setConnected(eventsConnected);
    listeners.add(l);
    return () => {
      listeners.delete(l);
    };
  }, []);

  return { events, connected };
}
