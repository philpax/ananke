import { useEffect, useState } from "preact/hooks";

import { api } from "./client";
import type { SystemEvent } from "./contract";
import { useResource } from "./resource";
import { connectionStatus } from "./socket";

export type CapturedEvent = {
  readonly event: SystemEvent;
  readonly receivedAt: number;
};

export function useSystem() {
  const services = useResource("services", api.services);
  const devices = useResource("devices", api.devices);
  const info = useResource("info", api.info);
  const events = useResource("events", api.events);
  const [connected, setConnected] = useState(false);
  const [openedAt] = useState(Date.now);
  useEffect(() => connectionStatus(setConnected), []);
  return {
    services,
    devices,
    info,
    connected,
    events: (events.data?.events ?? []).map((event) => ({
      event,
      receivedAt:
        "at_ms" in event ? event.at_ms : (events.data?.since_ms ?? openedAt),
    })),
    capturedSince: events.data?.since_ms ?? openedAt,
  };
}
