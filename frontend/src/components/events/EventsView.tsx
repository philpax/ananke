// Events (`/events`): the daemon's event history over the shared time
// range, merged with the persisted restart log, newest first. The history
// is a live resource, so new events arrive on their own; the socket
// reconnects by itself, so the header shows its state rather than a
// reconnect button.
//
// Newest first means following new events needs no code: at the top, a
// new row appears above the others; scrolled down, the browser's scroll
// anchoring holds the rows being read in place.

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useLocation } from "react-router-dom";

import { useEventFeed, useRestarts } from "../../api/hooks.ts";
import type { LiveStatus } from "../../api/live/socket.ts";
import { useLiveStatus } from "../../api/live/useLive.ts";
import { formatTimestamp } from "../../util.ts";
import { EmptyState } from "../ui/feedback/EmptyState.tsx";
import { Message } from "../ui/feedback/Message.tsx";
import { TimeRangeSelect } from "../ui/form/TimeRangeSelect.tsx";
import { rangeSearch } from "../ui/form/timeRange.ts";
import { useResolvedRange, useTimeRange } from "../ui/form/useTimeRange.ts";
import { PageHeader } from "../ui/layout/PageHeader.tsx";
import { Chip } from "../ui/status/Chip.tsx";
import type { ChipTone } from "../ui/status/chipVariants.ts";
import { EventList } from "./EventList.tsx";
import { EventTypeFilter } from "./EventTypeFilter.tsx";
import { eventsInRange, type EventType } from "./eventLog.ts";

export function EventsView() {
  const { t } = useTranslation();
  const { search } = useLocation();
  const [range, setRange] = useTimeRange();
  const { since, until, end } = useResolvedRange(range);
  const [types, setTypes] = useState<ReadonlySet<EventType>>(NO_TYPES);
  const feed = useEventFeed();
  const restarts = useRestarts(undefined, since, until ?? end);
  const status = useLiveStatus();
  const liveSince = feed.data?.since_ms;
  // A rolling range's end lags the clock by up to a step; events newer
  // than it still belong.
  const events = eventsInRange(
    feed.data?.events ?? [],
    restarts.data ?? [],
    since,
    until ?? Number.MAX_SAFE_INTEGER,
    types,
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PageHeader
        title={t("events.title")}
        meta={[
          t("events.shown", { count: events.length }),
          liveSince === undefined
            ? null
            : t("events.liveSince", { time: formatTimestamp(liveSince) }),
        ]
          .filter((part) => part !== null)
          .join(" · ")}
        actions={
          <Chip tone={STATUS_TONE[status]}>{t(`events.status.${status}`)}</Chip>
        }
      />
      <div className="flex shrink-0 flex-col gap-2 border-b border-line bg-surface px-4 py-2">
        <TimeRangeSelect value={range} onChange={setRange} />
        <EventTypeFilter selected={types} onChange={setTypes} />
      </div>
      <div className="@container/events min-h-0 flex-1 overflow-auto bg-canvas">
        {liveSince !== undefined && since < liveSince && (
          <p className="border-b border-line-subtle px-4 py-1.5 text-xs text-muted">
            {t("events.olderThanLive", {
              time: formatTimestamp(liveSince),
            })}
          </p>
        )}
        {feed.error && (
          <Message kind="error" text={feed.error.message} className="px-4" />
        )}
        {restarts.error && (
          <Message
            kind="error"
            text={restarts.error.message}
            className="px-4"
          />
        )}
        {events.length > 0 ? (
          <EventList events={events} nowMs={end} range={rangeSearch(search)} />
        ) : feed.isPending ? (
          <Message kind="loading" className="px-4" />
        ) : (
          <EmptyState message={t("events.emptyState")} />
        )}
      </div>
    </div>
  );
}

const NO_TYPES: ReadonlySet<EventType> = new Set();

const STATUS_TONE: Readonly<Record<LiveStatus, ChipTone>> = {
  open: "ok",
  connecting: "neutral",
  reconnecting: "warn",
};
