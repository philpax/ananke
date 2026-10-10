import clsx from "clsx";

import type { Route } from "../layout/navigation";
import { time } from "../ui/format";
import { eventLabel, eventSummary, eventTone } from "./data";
import type { eventsInRange } from "./data";

export function EventRows({
  rows,
  all = false,
  onNavigate,
}: {
  readonly rows: ReturnType<typeof eventsInRange>;
  readonly all?: boolean;
  readonly onNavigate?: (route: Route) => void;
}) {
  const columns = all
    ? "grid-cols-[82px_168px_minmax(220px,min(38vw,420px))_minmax(180px,1fr)]"
    : "grid-cols-[82px_168px_minmax(0,1fr)]";
  return (
    <>
      {all && (
        <div
          class={clsx(
            `
              sticky top-0 z-1 grid gap-3.5 border-b border-line-strong
              bg-canvas py-2.5 text-xs tracking-[0.06em] text-muted uppercase
              max-[899px]:hidden
            `,
            columns,
          )}
        >
          <span>Time</span>
          <span>Type</span>
          <span>Service</span>
          <span>Summary</span>
        </div>
      )}
      {rows.map((row, index) => (
        <div
          key={`${row.at}:${index}`}
          class={clsx(
            `
              grid gap-x-3.5 gap-y-1.5 border-b border-line-subtle py-[9px]
              text-[13px]
              max-[899px]:grid-cols-1
            `,
            columns,
            all && `border-l-4 pl-2`,
            all &&
              (eventTone(row.event.type) === "danger"
                ? `border-l-danger`
                : eventTone(row.event.type) === "warn"
                  ? `border-l-warn`
                  : `border-l-transparent`),
          )}
        >
          <time
            dateTime={new Date(row.at).toISOString()}
            title={new Date(row.at).toLocaleString()}
            class="font-mono text-muted"
          >
            {time(row.at)}
          </time>
          <span class="flex items-start">
            <EventBadge type={row.event.type} />
          </span>
          {all && (
            <span class="font-mono wrap-anywhere text-ink">
              {"service" in row.event ? (
                <button
                  type="button"
                  class="
                    text-left
                    hover:text-accent
                  "
                  onClick={() =>
                    onNavigate?.({
                      section: "services",
                      service:
                        "service" in row.event ? row.event.service : undefined,
                    })
                  }
                >
                  {row.event.service}
                </button>
              ) : (
                "—"
              )}
            </span>
          )}
          <span class="wrap-anywhere text-body">{eventSummary(row.event)}</span>
        </div>
      ))}
    </>
  );
}

export function EventBadge({ type }: { readonly type: string }) {
  const tone = eventTone(type);
  return (
    <span
      title={type}
      class={clsx(
        "border px-[7px] py-px text-xs",
        tone === "ok"
          ? `border-ok-line text-ok`
          : tone === "warn"
            ? `border-warn-line text-warn`
            : tone === "danger"
              ? `border-danger-line text-danger`
              : `border-line text-body`,
      )}
    >
      {eventLabel(type)}
    </span>
  );
}
