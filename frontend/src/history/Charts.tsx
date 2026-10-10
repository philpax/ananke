import * as Plot from "@observablehq/plot";
import { useEffect, useRef, useState } from "preact/hooks";

import type { Range } from "../api/client";
import { number, time } from "../ui/format";
import type { ChartData } from "./data";

export function Charts({
  title,
  charts,
  range,
  fleet = false,
}: {
  readonly title: string;
  readonly charts: readonly ChartData[];
  readonly range: Range;
  readonly fleet?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(1);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0].contentRect.width;
      setColumns(w >= 1320 ? 3 : w >= 760 ? 2 : 1);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return (
    <section aria-label={title} class="min-w-0 border-t border-line pt-3">
      <h3 class="mb-5 text-[13px] leading-[1.35] font-semibold">{title}</h3>
      <div
        ref={ref}
        class="grid min-w-0 gap-2.5"
        style={{ gridTemplateColumns: `repeat(${columns},minmax(0,1fr))` }}
      >
        {charts.map((chart) => (
          <Chart
            key={chart.title}
            chart={chart}
            range={range}
            height={fleet ? 280 : 230}
          />
        ))}
      </div>
    </section>
  );
}

function Chart({
  chart,
  range,
  height,
}: {
  readonly chart: ChartData;
  readonly range: Range;
  readonly height: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let frame = 0;
    function draw() {
      if (!el) return;
      const marks: Plot.Markish[] = [
        Plot.ruleY([0], { stroke: "var(--line)" }),
      ];
      let max = 0;
      for (const series of chart.series) {
        const points = series.points.map((p) => ({
          at: new Date(p.at),
          value: p.value,
          title: `${series.label}: ${number(p.value, 3)} ${chart.unit}\n${time(p.at)}`,
        }));
        for (const p of points)
          if (p.value !== null) max = Math.max(max, p.value);
        marks.push(
          Plot.line(points, {
            x: "at",
            y: "value",
            stroke: series.color,
            strokeWidth: 2.2,
          }),
        );
        marks.push(
          Plot.dot(points, {
            x: "at",
            y: "value",
            fill: series.color,
            r: 2.5,
            title: "title",
            tip: {
              fill: "#202126",
              stroke: "#777985",
              fontSize: 13,
              textPadding: 9,
            },
          }),
        );
      }
      const ticks = [range.since, (range.since + range.until) / 2, range.until];
      const plot = Plot.plot({
        width: Math.max(1, el.clientWidth),
        height,
        marginLeft: 68,
        marginRight: 12,
        marginTop: 12,
        marginBottom: 34,
        style: {
          background: "transparent",
          color: "var(--text-body)",
          fontFamily: "Work Sans,system-ui,sans-serif",
          fontSize: "15px",
        },
        x: {
          type: "utc",
          domain: [new Date(range.since), new Date(range.until)],
          ticks: ticks.map((t) => new Date(t)),
          tickFormat: (value) => {
            const t = Number(value);
            if (range.preset)
              return t === range.until
                ? "now"
                : `${number((t - range.until) / 3_600_000, 1)}h`;
            return time(t).slice(0, 5);
          },
          label: null,
          tickSize: 0,
        },
        y: {
          domain: [0, max > 0 ? max * 1.12 : 1],
          nice: true,
          ticks: 4,
          grid: true,
          label: null,
        },
        marks,
      });
      plot.setAttribute(
        "aria-label",
        `${chart.title} over the selected time range`,
      );
      el.replaceChildren(plot);
    }
    const queue = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(draw);
    };
    const observer = new ResizeObserver(queue);
    observer.observe(el);
    queue();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
      el.replaceChildren();
    };
  }, [chart, range, height]);
  const empty = !chart.series.some((s) =>
    s.points.some((p) => p.value !== null),
  );
  return (
    <div
      class="
        min-w-0 overflow-hidden border border-line bg-surface px-3 pt-3 pb-2
      "
    >
      <div class="flex min-h-[22px] flex-wrap items-baseline gap-2 text-body">
        <b class="text-base font-semibold text-ink">{chart.title}</b>
        <span class="font-mono text-[13px] text-muted">{chart.unit}</span>
      </div>
      {(chart.series.length > 1 || chart.series[0]?.label !== chart.title) && (
        <div
          class="
            flex min-h-[26px] flex-wrap items-center gap-x-3.5 gap-y-1 text-sm
          "
        >
          {chart.series.map((s) => (
            <span key={s.label} style={{ color: s.color }}>
              ━ {s.label}
            </span>
          ))}
        </div>
      )}
      <div
        ref={ref}
        class="min-w-0 scheme-dark"
        style={{ height: `${height}px` }}
      />
      {empty && <p class="text-xs text-muted">No samples in this range.</p>}
    </div>
  );
}
