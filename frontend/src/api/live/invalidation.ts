// Refreshes the queries that are not live resources when an event says they
// changed: the config editor after a reload, and the restart history after
// an auto-restart. Everything else updates through its own subscription.

import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { DaemonEvent } from "../client.ts";
import { eventsQuery } from "./resources.ts";
import { useLive } from "./useLive.ts";

export function useEventInvalidation(): void {
  const qc = useQueryClient();
  const events = useLive(eventsQuery).data?.events;
  const newestSeen = useRef<string | null>(null);

  useEffect(() => {
    if (events === undefined) return;
    const fresh = eventsAfter(events, newestSeen.current);
    const newest = events.at(-1);
    newestSeen.current = newest === undefined ? "" : JSON.stringify(newest);
    if (fresh.some((e) => e.type === "config_reloaded")) {
      void qc.invalidateQueries({ queryKey: ["config"] });
    }
    if (fresh.some((e) => e.type === "auto_restarted")) {
      void qc.invalidateQueries({ queryKey: ["restarts"] });
    }
  }, [events, qc]);
}

/// The events newer than the one serialised as `newest`. The history only
/// appends (and evicts from the front), so they are the ones after its last
/// occurrence. `null` means nothing has been seen yet, so the backfill is not
/// news; an empty string means the history was empty, so everything is.
export function eventsAfter(
  events: readonly DaemonEvent[],
  newest: string | null,
): readonly DaemonEvent[] {
  if (newest === null) return [];
  if (newest === "") return events;
  for (let i = events.length - 1; i >= 0; i--) {
    if (JSON.stringify(events[i]) === newest) return events.slice(i + 1);
  }
  return events;
}
