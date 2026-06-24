// Event feed — live, virtualised feed of system events from the
// /api/events WebSocket. Shows timestamp, event type, service name,
// and key fields as one-line summaries.

import { useState } from "react";
import { useEvents, type SystemEvent } from "../../api/events.ts";
import { formatTimestamp } from "../../util.ts";

const EVENT_TYPES = [
  "state_changed",
  "allocation_changed",
  "config_reloaded",
  "estimator_drift",
] as const;

export function EventsView() {
  const { events, connected } = useEvents();
  const [paused, setPaused] = useState(false);
  const [typeFilter, setTypeFilter] = useState<Set<string>>(
    new Set(EVENT_TYPES),
  );

  const filtered = paused
    ? events
    : events.filter((e) => typeFilter.size === 0 || typeFilter.has(e.type));

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
      <div className="flex items-center gap-2 border-b border-border-default px-3 py-2">
        <h1 className="text-base font-medium text-primary">Events</h1>
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
          <button
            onClick={() => setPaused((p) => !p)}
            className={`rounded-sm px-2 py-0.5 transition-colors ${
              paused ? "text-warning" : "text-tertiary hover:text-secondary"
            }`}
          >
            {paused ? "paused" : "pause"}
          </button>
          <span className="font-mono">{filtered.length}</span>
        </div>
      </div>

      {/* Event list */}
      <div className="flex-1 overflow-auto">
        {filtered.length === 0 ? (
          <div className="flex min-h-[200px] items-center justify-center text-sm text-tertiary">
            No events yet.
          </div>
        ) : (
          [...filtered]
            .reverse()
            .map((event, i) => <EventRow key={i} event={event} />)
        )}
      </div>
    </div>
  );
}

function EventRow({ event }: { event: SystemEvent }) {
  const variant = EVENT_VARIANTS[event.type] ?? "neutral";
  return (
    <div className="flex items-center gap-3 border-b border-border-default px-3 py-1.5 hover:bg-elevated">
      <span className="shrink-0 font-mono text-xs text-tertiary">
        {formatTimestamp(event.timestamp)}
      </span>
      <span
        className={`shrink-0 rounded-sm px-1.5 py-0.5 text-xs font-medium ${variant}`}
      >
        {event.type}
      </span>
      {event.service && (
        <span className="shrink-0 font-mono text-xs text-primary">
          {event.service}
        </span>
      )}
      <span className="truncate text-xs text-secondary">
        {summarizeEvent(event)}
      </span>
    </div>
  );
}

const EVENT_VARIANTS: Record<string, string> = {
  state_changed: "bg-success/15 text-success",
  allocation_changed: "bg-accent/15 text-accent",
  config_reloaded: "bg-warning/15 text-warning",
  estimator_drift: "bg-vision/15 text-vision",
};

function summarizeEvent(event: SystemEvent): string {
  switch (event.type) {
    case "state_changed": {
      const from = event.from ?? "?";
      const to = event.to ?? "?";
      return `${from} → ${to}`;
    }
    case "allocation_changed":
      return "device allocation shifted";
    case "config_reloaded":
      return "configuration reloaded from disk";
    case "estimator_drift": {
      const mean = event.rolling_mean;
      const samples = event.rolling_samples;
      if (typeof mean === "number" && typeof samples === "number") {
        return `correction ${mean.toFixed(3)} (n=${samples})`;
      }
      return "estimate correction updated";
    }
    default:
      return "";
  }
}
