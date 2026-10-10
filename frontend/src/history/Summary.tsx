import clsx from "clsx";

import type { Bucket } from "../api/contract";
import { number } from "../ui/format";
import { totals } from "./data";

export function Summary({
  buckets,
  fleet = false,
  unavailable = false,
}: {
  readonly buckets: readonly Bucket[];
  readonly fleet?: boolean;
  readonly unavailable?: boolean;
}) {
  const t = totals(buckets);
  const stats = fleet
    ? ([
        ["Requests", t.requests],
        ["Errors", t.errors],
        ["Tokens", t.tokens],
        [
          "Avg latency",
          t.latency === null ? null : `${number(t.latency, 0)} ms`,
        ],
        ["Avg in tok/s", t.input],
        ["Avg out tok/s", t.output],
      ] as const)
    : ([
        ["Requests", t.requests],
        [
          "Avg latency",
          t.latency === null ? null : `${number(t.latency, 0)} ms`,
        ],
        ["Out tok/s", t.output],
        ["Errors", t.errors],
      ] as const);
  return (
    <div
      class={clsx(
        "grid gap-y-3",
        fleet
          ? `grid-cols-[repeat(auto-fit,minmax(140px,1fr))] gap-x-7`
          : `grid-cols-[repeat(auto-fit,minmax(120px,1fr))] gap-x-6`,
      )}
    >
      {stats.map(([label, value]) => (
        <div key={label} class="flex min-w-0 flex-col gap-1">
          <span class="text-xs tracking-[0.05em] text-muted uppercase">
            {label}
          </span>
          <span
            class={clsx(
              `text-[19px] font-semibold tabular-nums`,
              fleet && `text-xl`,
              fleet && label === "Errors" ? `text-danger` : `text-ink`,
            )}
          >
            {unavailable
              ? "—"
              : typeof value === "string"
                ? value
                : number(value)}
          </span>
        </div>
      ))}
    </div>
  );
}
