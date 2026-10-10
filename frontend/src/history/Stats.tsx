import type { Range } from "../api/client";
import { api, rangeKey } from "../api/client";
import { useResource } from "../api/resource";
import { Message } from "../ui/Message";
import { PageHeader } from "../ui/PageHeader";
import { number } from "../ui/format";
import { Charts } from "./Charts";
import { RangeControl } from "./RangeControl";
import { Summary } from "./Summary";
import { activityCharts, serviceSpeeds } from "./data";

export function Stats({
  range,
  onRange,
}: {
  readonly range: Range;
  readonly onRange: (range: Range) => void;
}) {
  const key = rangeKey(range);
  const metrics = useResource(key, api.metrics(range));
  const restarts = useResource(key, api.restarts(range));
  const samples = useResource(key, api.samples(range));
  const buckets = metrics.data?.buckets ?? [];
  const speeds = serviceSpeeds(buckets);
  const max = Math.max(0, ...speeds.map((s) => s.value ?? 0));
  const charts = activityCharts(
    buckets,
    restarts.data?.restarts ?? [],
    samples.data?.samples ?? [],
    range,
    true,
  );
  return (
    <main
      data-screen-label="Stats"
      class="min-w-0 flex-1 overflow-auto bg-canvas"
    >
      <PageHeader title="Stats" meta="Fleet activity" />
      <div
        aria-label="Fleet history"
        class="
          mx-6 mt-4 mb-9 flex min-w-0 flex-col gap-[18px]
          max-[899px]:mx-3
        "
      >
        <h2 class="text-[15px] font-semibold">History</h2>
        <RangeControl range={range} onChange={onRange} />
        {(metrics.error || restarts.error || samples.error) && (
          <Message error={metrics.error ?? restarts.error ?? samples.error} />
        )}
        <section
          aria-label="Fleet totals"
          class="border-b border-line pt-0.5 pb-[18px]"
        >
          <Summary
            buckets={buckets}
            fleet
            unavailable={metrics.loading || !!metrics.error}
          />
        </section>
        <Charts title="Fleet activity" charts={charts} range={range} fleet />
        <section class="border-t border-line">
          <h2 class="mb-[9px] text-[13px] leading-[1.35] font-semibold">
            Output tokens per second, by service
          </h2>
          {speeds.map((s) => (
            <div
              key={s.name}
              class="
                grid grid-cols-[280px_minmax(0,1fr)_70px] items-center gap-x-3.5
                gap-y-1.5 border-b border-line-subtle py-2
                max-[899px]:grid-cols-1
              "
            >
              <span class="font-mono text-[12.5px] leading-[1.4] wrap-anywhere">
                {s.name}
              </span>
              <div
                role="img"
                aria-label={`${s.name}: ${number(s.value)} output tokens per second`}
                class="h-3.5 min-w-0 border border-line-strong bg-canvas"
              >
                <div
                  class="h-full bg-accent-data"
                  style={{
                    width: `${max && s.value != null ? (s.value / max) * 100 : 0}%`,
                  }}
                />
              </div>
              <span class="text-right text-[12.5px] font-semibold tabular-nums">
                {number(s.value)}
              </span>
            </div>
          ))}
          {!speeds.length && (
            <Message
              loading={metrics.loading}
              empty="No service metrics in this range."
            />
          )}
          <div class="flex justify-between pt-2 text-xs text-muted tabular-nums">
            <span>0</span>
            <span>{number(max)} tok/s</span>
          </div>
        </section>
      </div>
    </main>
  );
}
