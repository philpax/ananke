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

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { api } from "./client.ts";
import {
  handleEvent,
  refresh as refreshStore,
  startPolling,
  stopPolling,
} from "./systemStore.ts";

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

// Set by useEventsConnection while the socket is live. Calling
// reconnectEvents() closes the current socket and immediately opens
// a fresh one — no 2-second delay.
let reconnectTrigger: (() => void) | null = null;

export function reconnectEvents(): void {
  reconnectTrigger?.();
}

// --- Hooks ---

/// Opens the events WebSocket on mount and drives query invalidation.
/// Call once in App. Does not return event data.
export function useEventsConnection(): void {
  const qc = useQueryClient();
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let shouldReconnect = true;

    const connect = () => {
      if (!shouldReconnect) return;
      socket = new WebSocket(api.eventsStreamUrl());
      socket.onopen = () => {
        eventsConnected = true;
        for (const l of listeners) l();
        stopPolling();
        void refreshStore();
        void qc.invalidateQueries({ queryKey: ["config"] });
        void qc.invalidateQueries({ queryKey: ["service-detail"] });
      };
      socket.onclose = () => {
        eventsConnected = false;
        for (const l of listeners) l();
        startPolling();
        if (!shouldReconnect) return;
        reconnectTimer = window.setTimeout(connect, 2_000);
      };
      socket.onerror = () => {
        // onerror doesn't always trigger onclose promptly, so close
        // the socket explicitly to force the reconnect cycle.
        eventsConnected = false;
        for (const l of listeners) l();
        socket?.close();
      };
      socket.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        let event: SystemEvent;
        try {
          event = JSON.parse(ev.data) as SystemEvent;
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
      };
    };
    connect();

    reconnectTrigger = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (socket) {
        socket.onclose = null;
        socket.onerror = null;
        socket.onmessage = null;
        socket.close();
        socket = null;
      }
      connect();
    };

    return () => {
      shouldReconnect = false;
      cancelled.current = true;
      reconnectTrigger = null;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [qc]);
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
