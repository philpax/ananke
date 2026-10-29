// The events view's rows: the live event history merged with the persisted
// restart log, filtered to the shared time range and the chosen types, and
// how each row is toned. Pure, so the merging and filtering are testable
// without a render.

import type { DaemonEvent, ServiceRestartEntry } from "../../api/client.ts";
import type { Tone } from "../ui/tone.ts";
import { stateTone } from "../ui/status/stateTone.ts";

export type EventType = DaemonEvent["type"];

export const EVENT_TYPES = [
  "state_changed",
  "allocation_changed",
  "config_reloaded",
  "estimator_drift",
  "auto_restarted",
  "reservation_placed",
  "reservation_removed",
] as const satisfies readonly EventType[];

/// The live history plus every persisted restart it does not already hold.
/// The restart log reaches back past the daemon's in-memory window, so an
/// old range still shows why a service was restarted. Order is not kept.
export function mergeRestarts(
  events: readonly DaemonEvent[],
  restarts: readonly ServiceRestartEntry[],
): DaemonEvent[] {
  const merged = [...events];
  const seen = new Set(
    events
      .filter((e) => e.type === "auto_restarted")
      .map((e) => restartKey(e.service, e.at_ms, e.trigger)),
  );
  for (const r of restarts) {
    const key = restartKey(r.service, r.at_ms, r.trigger);
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push({
      type: "auto_restarted",
      service: r.service,
      trigger: r.trigger,
      detail: r.detail,
      at_ms: r.at_ms,
    });
  }
  return merged;
}

/// The rows for `[since, until]`, newest first. An empty `types` set means
/// no type filter, so clearing every toggle never hides everything.
export function eventsInRange(
  events: readonly DaemonEvent[],
  restarts: readonly ServiceRestartEntry[],
  since: number,
  until: number,
  types: ReadonlySet<EventType>,
): DaemonEvent[] {
  return mergeRestarts(events, restarts)
    .filter(
      (e) =>
        e.at_ms >= since &&
        e.at_ms <= until &&
        (types.size === 0 || types.has(e.type)),
    )
    .sort((a, b) => b.at_ms - a.at_ms);
}

/// `types` with `type` toggled in or out.
export function toggleType(
  types: ReadonlySet<EventType>,
  type: EventType,
): ReadonlySet<EventType> {
  const next = new Set(types);
  if (next.has(type)) next.delete(type);
  else next.add(type);
  return next;
}

/// How alarming an event is. A state change takes the tone of the state it
/// entered, so a failure reads as one; drift and auto-restarts mean the
/// daemon corrected something on its own.
export function eventTone(event: DaemonEvent): Tone {
  switch (event.type) {
    case "state_changed":
      return stateTone(event.to, event.disabled_reason);
    case "estimator_drift":
    case "auto_restarted":
      return "warn";
    case "allocation_changed":
    case "config_reloaded":
    case "reservation_placed":
    case "reservation_removed":
      return "neutral";
    default: {
      const unreachable: never = event;
      return unreachable;
    }
  }
}

/// The one service an event is about, or `null` for a fleet-wide one.
export function eventService(event: DaemonEvent): string | null {
  return "service" in event ? event.service : null;
}

/// The `events.types.*` key for a type.
export function eventTypeLabelKey(type: EventType): string {
  return `events.types.${type}`;
}

function restartKey(service: string, atMs: number, trigger: string): string {
  return `${service}:${atMs}:${trigger}`;
}
