import clsx from "clsx";
import { useState } from "preact/hooks";

import type { Range } from "../api/client";
import { api, rangeKey } from "../api/client";
import { useResource } from "../api/resource";
import type { CapturedEvent } from "../api/system";
import { RangeControl } from "../history/RangeControl";
import type { Route } from "../layout/navigation";
import { Message } from "../ui/Message";
import { PageHeader } from "../ui/PageHeader";
import { time } from "../ui/format";
import { EventRows } from "./EventRows";
import { eventLabel, eventsInRange, eventTone, eventTypes } from "./data";
import type { EventType } from "./data";

export function Events({
  range,
  onRange,
  captured,
  capturedSince,
  connected,
  onNavigate,
}: {
  readonly range: Range;
  readonly onRange: (r: Range) => void;
  readonly captured: readonly CapturedEvent[];
  readonly capturedSince: number;
  readonly connected: boolean;
  readonly onNavigate: (route: Route) => void;
}) {
  const [type, setType] = useState<EventType>("all");
  const restarts = useResource(rangeKey(range), api.restarts(range));
  const rows = eventsInRange(
    captured,
    restarts.data?.restarts ?? [],
    range,
    type,
  );
  return (
    <main
      data-screen-label="Events"
      class="flex min-w-0 flex-1 flex-col bg-canvas"
    >
      <PageHeader
        title="Events"
        meta={`${rows.length} shown · live events since ${time(capturedSince)}${connected ? "" : " · reconnecting"}`}
      />
      <div
        class="
          flex shrink-0 flex-col gap-2 border-b border-line bg-surface px-4 pt-2
          pb-2.5
        "
      >
        <RangeControl range={range} onChange={onRange} />
        <div
          role="group"
          aria-label="Filter by event type"
          class="flex flex-wrap gap-1"
        >
          {eventTypes.map((t) => (
            <button
              key={t}
              type="button"
              aria-pressed={type === t}
              class={clsx(
                `h-[30px] border px-2.5 font-mono text-[12.5px]`,
                type === t
                  ? `border-accent bg-selected text-accent-text`
                  : eventTone(t) === "ok"
                    ? `border-ok-line text-ok`
                    : eventTone(t) === "warn"
                      ? `border-warn-line text-warn`
                      : eventTone(t) === "danger"
                        ? `border-danger-line text-danger`
                        : `border-line-strong text-body`,
              )}
              onClick={() => setType(t)}
            >
              {eventLabel(t)}
            </button>
          ))}
        </div>
      </div>
      <div class="min-h-0 flex-1 overflow-auto px-4 pb-8">
        <EventRows rows={rows} all onNavigate={onNavigate} />
        {restarts.error && <Message error={restarts.error} />}{" "}
        {!rows.length && (
          <Message
            loading={restarts.loading}
            empty="No captured events or recorded restarts in this range."
          />
        )}
      </div>
    </main>
  );
}
