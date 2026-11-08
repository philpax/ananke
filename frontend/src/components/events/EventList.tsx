// The events as rows, newest first. Wide, the list is a table: a sticky
// column header over time, type, service, and summary columns. Narrow,
// each row stacks: time, type, and service on one line, the summary
// under them. The width is the list's own, so this is a container query.
// A left rail marks how alarming each event is.

import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import type { DaemonEvent } from "../../api/client.ts";
import { formatTimestamp } from "../../util.ts";
import { cn } from "../ui/cn.ts";
import { ColumnHeader } from "../ui/layout/ColumnHeader.tsx";
import { Chip } from "../ui/status/Chip.tsx";
import { toneRail } from "../ui/tone.ts";
import { eventService, eventTone, eventTypeLabelKey } from "./eventLog.ts";
import { summarizeEvent } from "./eventSummary.ts";

export function EventList({
  events,
  nowMs,
  range,
}: {
  readonly events: readonly DaemonEvent[];
  readonly nowMs: number;
  /// The current range's query, carried into service links.
  readonly range: string;
}) {
  const { t } = useTranslation();
  return (
    <>
      <ColumnHeader
        className={cn(
          "sticky top-0 z-10 hidden border-l-4 border-l-transparent @min-events-cols/events:grid",
          columnsClassName,
        )}
      >
        <span>{t("events.time")}</span>
        <span>{t("events.type")}</span>
        <span>{t("events.service")}</span>
        <span>{t("events.summary")}</span>
      </ColumnHeader>
      <ol>
        {events.map((event, i) => (
          <EventRow
            key={`${event.at_ms}-${event.type}-${i}`}
            event={event}
            nowMs={nowMs}
            range={range}
          />
        ))}
      </ol>
    </>
  );
}

const columnsClassName =
  "@min-events-cols/events:grid-cols-[8rem_9.5rem_minmax(8rem,16rem)_minmax(0,1fr)]";

function EventRow({
  event,
  nowMs,
  range,
}: {
  readonly event: DaemonEvent;
  readonly nowMs: number;
  readonly range: string;
}) {
  const { t } = useTranslation();
  const service = eventService(event);
  return (
    <li
      className={cn(
        "flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-line-subtle px-4 py-2 text-sm @min-events-cols/events:grid @min-events-cols/events:gap-y-0",
        columnsClassName,
        toneRail({ tone: eventTone(event) }),
      )}
    >
      <time
        dateTime={new Date(event.at_ms).toISOString()}
        title={formatTimestamp(event.at_ms)}
        className="shrink-0 font-mono text-xs text-muted tabular-nums"
      >
        {clockTime(event.at_ms, nowMs)}
      </time>
      <span className="shrink-0">
        <Chip>{t(eventTypeLabelKey(event.type))}</Chip>
      </span>
      <span className="min-w-0 font-mono text-xs wrap-anywhere">
        {service === null ? (
          <span className="text-faint">—</span>
        ) : (
          <Link
            to={`/services/${encodeURIComponent(service)}${range}`}
            className="text-ink hover:text-accent-text hover:underline"
          >
            {service}
          </Link>
        )}
      </span>
      <span className="min-w-0 basis-full wrap-anywhere text-body @min-events-cols/events:basis-auto">
        {summarizeEvent(event, t)}
      </span>
    </li>
  );
}

/// The time of day, with the date when the event was not today.
function clockTime(atMs: number, nowMs: number): string {
  const at = new Date(atMs);
  const time = at.toLocaleTimeString(undefined, { hour12: false });
  return at.toDateString() === new Date(nowMs).toDateString()
    ? time
    : `${at.toLocaleDateString(undefined, { month: "short", day: "numeric" })} ${time}`;
}
