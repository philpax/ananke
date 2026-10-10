import type { Range } from "../api/client";
import type { Restart, SystemEvent } from "../api/contract";
import type { CapturedEvent } from "../api/system";
import { memory, stateLabel } from "../ui/format";

export const eventTypes = [
  "all",
  "state_changed",
  "allocation_changed",
  "config_reloaded",
  "estimator_drift",
  "auto_restarted",
  "overflow",
] as const;
export type EventType = (typeof eventTypes)[number];

export function eventLabel(type: string) {
  return (
    {
      all: "All",
      state_changed: "State changed",
      allocation_changed: "Allocation changed",
      config_reloaded: "Config reloaded",
      estimator_drift: "Estimate drift",
      auto_restarted: "Automatically restarted",
      overflow: "Capacity overflow",
    }[type] ?? type.replaceAll("_", " ")
  );
}

export function eventTone(type: string) {
  return type === "overflow"
    ? "danger"
    : type === "estimator_drift" || type === "auto_restarted"
      ? "warn"
      : type === "state_changed"
        ? "ok"
        : "neutral";
}

export function eventSummary(event: SystemEvent): string {
  switch (event.type) {
    case "state_changed":
      return `${stateLabel(event.from)} → ${stateLabel(event.to)}`;
    case "allocation_changed":
      return (
        Object.entries(event.reservations)
          .map(([device, bytes]) => `${memory(bytes)} on ${device}`)
          .join(" · ") || "Reservations released"
      );
    case "config_reloaded":
      return event.changed_services.length
        ? `Configuration reloaded · ${event.changed_services.length} services changed`
        : "Configuration reloaded";
    case "estimator_drift":
      return `${event.class} correction ${event.rolling_mean.toFixed(3)}×`;
    case "auto_restarted":
      return `${event.trigger.replaceAll("_", " ")} · ${event.detail}`;
    case "overflow":
      return `${event.dropped} events dropped from the buffer`;
  }
}

export function eventsInRange(
  captured: readonly CapturedEvent[],
  restarts: readonly Restart[],
  range: Range,
  type = "all",
  service?: string,
) {
  const rows = [...captured];
  for (const r of restarts)
    if (
      !rows.some(
        (c) =>
          c.event.type === "auto_restarted" &&
          c.event.service === r.service &&
          c.event.at_ms === r.at_ms,
      )
    )
      rows.push({
        event: {
          type: "auto_restarted",
          service: r.service,
          at_ms: r.at_ms,
          trigger: r.trigger,
          detail: r.detail,
        },
        receivedAt: r.at_ms,
      });
  return rows
    .map((row) => ({
      ...row,
      at: "at_ms" in row.event ? row.event.at_ms : row.receivedAt,
    }))
    .filter(
      (row) =>
        row.at >= range.since &&
        row.at <= (range.preset !== null ? Date.now() : range.until) &&
        (type === "all" || row.event.type === type) &&
        (!service || ("service" in row.event && row.event.service === service)),
    )
    .sort((a, b) => b.at - a.at);
}
