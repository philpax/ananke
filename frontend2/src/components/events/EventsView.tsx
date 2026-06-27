// Event feed — live feed of system events from the /api/events
// WebSocket. Shows timestamp, event type, service name, and key
// fields as one-line summaries in a table layout.

import { useState } from "react";
import { useEventFeed, type SystemEvent } from "../../api/events.ts";

const EVENT_TYPES = [
  "state_changed",
  "allocation_changed",
  "config_reloaded",
  "estimator_drift",
  "overflow",
] as const;

export function EventsView() {
  const { events, connected } = useEventFeed();
  const [typeFilter, setTypeFilter] = useState<Set<string>>(
    new Set(EVENT_TYPES),
  );

  const filtered = events.filter(
    (e) => typeFilter.size === 0 || typeFilter.has(e.type),
  );

  function toggleType(type: string) {
    setTypeFilter((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex h-14 shrink-0 items-center gap-2 border-b border-border-default px-4">
        <h1 className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-primary">
          Events
        </h1>
        <div className="mx-2 h-3 w-px bg-border-default" />
        {EVENT_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => toggleType(type)}
            className={`rounded-sm px-2 py-0.5 text-xs transition-colors ${
              typeFilter.has(type)
                ? "bg-elevated text-primary"
                : "text-tertiary hover:text-secondary"
            }`}
          >
            {type}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2 text-xs text-tertiary">
          {connected ? (
            <span className="text-success">connected</span>
          ) : (
            <span className="text-danger">disconnected</span>
          )}
        </div>
      </div>

      {/* Event table */}
      <div className="flex-1 overflow-auto">
        {filtered.length === 0 ? (
          <div className="flex min-h-[200px] items-center justify-center text-sm text-tertiary">
            No events yet.
          </div>
        ) : (
          <table className="w-full">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-border-default text-left">
                <th className="w-28 px-4 py-1.5 text-xs font-medium text-tertiary">
                  Time
                </th>
                <th className="w-40 px-4 py-1.5 text-xs font-medium text-tertiary">
                  Type
                </th>
                <th className="w-40 px-4 py-1.5 text-xs font-medium text-tertiary">
                  Service
                </th>
                <th className="px-4 py-1.5 text-xs font-medium text-tertiary">
                  Summary
                </th>
              </tr>
            </thead>
            <tbody>
              {[...filtered].reverse().map((event, i) => (
                <EventRow key={i} event={event} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function EventRow({ event }: { event: SystemEvent }) {
  const variant = EVENT_VARIANTS[event.type] ?? "neutral";
  return (
    <tr className="border-b border-border-default/50 text-xs hover:bg-elevated/60">
      <td className="px-4 py-1.5 font-mono text-tertiary">
        {formatEventTime(event.at_ms)}
      </td>
      <td className="px-4 py-1.5">
        <span
          className={`inline-block rounded-[3px] px-1.5 py-0.5 font-mono text-[0.625rem] font-medium uppercase tracking-[0.08em] ring-1 ring-inset ${variant}`}
        >
          {event.type}
        </span>
      </td>
      <td className="px-4 py-1.5 font-mono text-primary">
        {event.service ?? "—"}
      </td>
      <td className="px-4 py-1.5 text-secondary">{summarizeEvent(event)}</td>
    </tr>
  );
}

const EVENT_VARIANTS: Record<string, string> = {
  state_changed: "bg-success/12 text-success ring-success/25",
  allocation_changed: "bg-accent/12 text-accent ring-accent/25",
  config_reloaded: "bg-warning/12 text-warning ring-warning/25",
  estimator_drift: "bg-vision/12 text-vision ring-vision/25",
  overflow: "bg-danger/12 text-danger ring-danger/25",
};

function summarizeEvent(event: SystemEvent): string {
  switch (event.type) {
    case "state_changed": {
      const from = event.from ?? "?";
      const to = event.to ?? "?";
      return `${from} \u2192 ${to}`;
    }
    case "allocation_changed":
      return "device allocation shifted";
    case "config_reloaded":
      return "configuration reloaded from disk";
    case "estimator_drift": {
      const mean = event.rolling_mean;
      if (typeof mean === "number") {
        return `correction ${mean.toFixed(3)}`;
      }
      return "estimate correction updated";
    }
    case "overflow": {
      const dropped = event.dropped;
      return typeof dropped === "number"
        ? `${dropped} events dropped`
        : "events dropped";
    }
    default:
      return JSON.stringify(
        Object.fromEntries(
          Object.entries(event).filter(
            ([k]) => k !== "type" && k !== "service" && k !== "at_ms",
          ),
        ),
      );
  }
}

function formatEventTime(ts: number | undefined): string {
  if (!ts || !Number.isFinite(ts)) return "\u2014";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
