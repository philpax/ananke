// Event feed — live feed of system events from the /api/events
// WebSocket. Shows timestamp, event type, service name, and key
// fields as one-line summaries in a table layout. Oldest events at
// the top, newest at the bottom; autoscrolls to the bottom when new
// events arrive unless the user has scrolled up.

import { useEffect, useRef, useState } from "react";
import {
  reconnectEvents,
  useEventFeed,
  type SystemEvent,
} from "../../api/events.ts";
import { formatTimestamp } from "../../util.ts";
import { ViewHeader } from "../ui/ViewHeader.tsx";
import { Badge, type BadgeVariant } from "../ui/Badge.tsx";
import { EmptyState } from "../ui/EmptyState.tsx";

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

  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    autoScrollRef.current = atBottom;
  }

  useEffect(() => {
    if (scrollRef.current && autoScrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events.length]);

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <ViewHeader className="gap-2">
        <h1 className="eyebrow !text-primary">Events</h1>
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
            onClick={reconnectEvents}
            className="rounded-sm px-2 py-0.5 text-tertiary transition-colors hover:text-secondary"
            title="Force reconnect"
          >
            ↻
          </button>
        </div>
      </ViewHeader>

      {/* Event table */}
      <div ref={scrollRef} onScroll={onScroll} className="flex-1 overflow-auto">
        {filtered.length === 0 ? (
          <EmptyState message="No events yet." />
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
              {filtered.map((event, i) => (
                <EventRow
                  key={`${event.at_ms ?? i}-${event.type}`}
                  event={event}
                />
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
        {event.at_ms != null ? formatTimestamp(event.at_ms) : "\u2014"}
      </td>
      <td className="px-4 py-1.5">
        <Badge variant={variant}>{event.type}</Badge>
      </td>
      <td className="px-4 py-1.5 font-mono text-primary">
        {event.service ?? "—"}
      </td>
      <td className="px-4 py-1.5 text-secondary">{summarizeEvent(event)}</td>
    </tr>
  );
}

const EVENT_VARIANTS: Record<string, BadgeVariant> = {
  state_changed: "success",
  allocation_changed: "accent",
  config_reloaded: "warning",
  estimator_drift: "vision",
  overflow: "danger",
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
