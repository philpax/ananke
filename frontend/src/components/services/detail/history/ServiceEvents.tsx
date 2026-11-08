// The newest events about one service within the shared range: state
// changes (with why it was disabled), drift corrections, auto-restarts,
// and reloads that touched it.

import { useTranslation } from "react-i18next";
import { Link, useLocation } from "react-router-dom";

import { useEventFeed, useRestarts } from "../../../../api/hooks.ts";
import { formatTimestamp, relativeTime } from "../../../../util.ts";
import { summarizeEvent } from "../../../events/eventSummary.ts";
import { EmptyState } from "../../../ui/feedback/EmptyState.tsx";
import { rangeSearch } from "../../../ui/form/timeRange.ts";
import type { ResolvedRange } from "../../../ui/form/useTimeRange.ts";
import { Panel } from "../../../ui/layout/Panel.tsx";
import { serviceEvents } from "./serviceEvents.ts";

const SHOWN = 20;

export function ServiceEvents({
  name,
  resolved,
}: {
  readonly name: string;
  readonly resolved: ResolvedRange;
}) {
  const { t } = useTranslation();
  const { search } = useLocation();
  const feed = useEventFeed();
  const { since, until, end } = resolved;
  const restarts = useRestarts(name, since, until ?? end);
  // A rolling range's end lags the clock by up to a step; events newer
  // than it still belong.
  const upper = until ?? Number.MAX_SAFE_INTEGER;
  const events = serviceEvents(
    feed.data?.events ?? [],
    restarts.data ?? [],
    name,
    since,
    upper,
  ).slice(0, SHOWN);

  return (
    <Panel
      heading={t("services.detail.recentEvents")}
      actions={
        <Link
          to={`/events${rangeSearch(search)}`}
          className="text-xs text-accent-text hover:underline"
        >
          {t("services.detail.allEvents")}
        </Link>
      }
      padding="none"
    >
      {events.length === 0 ? (
        <EmptyState message={t("services.detail.noEvents")} />
      ) : (
        <ol className="divide-y divide-line-subtle">
          {events.map((event, i) => (
            <li
              key={`${event.at_ms}-${event.type}-${i}`}
              className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-4 py-1.5 text-sm"
            >
              <time
                dateTime={new Date(event.at_ms).toISOString()}
                title={formatTimestamp(event.at_ms)}
                className="w-20 shrink-0 font-mono text-xs text-muted tabular-nums"
              >
                {relativeTime(event.at_ms)}
              </time>
              <span className="min-w-0 wrap-anywhere text-body">
                {summarizeEvent(event, t)}
              </span>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}
