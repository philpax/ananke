// uPlot wrapper that handles creation, data updates, and resize without
// recreating the chart. Series take token colours, resolved against the
// live theme when the plot is built; a theme change rebuilds it. Renders
// its own legend below the chart: series labels on top, values on hover.

import { useEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import { zeroFillGaps } from "../../../util.ts";
import { cn } from "../cn.ts";
import { useTheme } from "../theme/useTheme.ts";
import { resolveTokenColors, resolveTokenValue } from "./resolveColor.ts";
import { fillToken, seriesBgClassName, type SeriesColor } from "./series.ts";
import { compactTick } from "./ticks.ts";

export type ChartSeries = {
  readonly label: string;
  readonly color: SeriesColor;
  /// Shade the area under the line with the colour's fill token.
  readonly fill?: boolean;
  readonly width?: number;
  readonly unit?: string;
  /// Draw a continuous line across null gaps. Defaults to `true`. Set
  /// `false` for sparse metrics (e.g. a decode rate that is undefined when
  /// nothing is generating) so the line breaks instead of interpolating a
  /// misleading flat value across the gap.
  readonly spanGaps?: boolean;
};

export type ChartProps = {
  readonly data: (number | null)[][];
  readonly series: readonly ChartSeries[];
  readonly height?: number;
  readonly xMin?: number;
  readonly xMax?: number;
};

export function Chart({ data, series, height = 160, xMin, xMax }: ChartProps) {
  const { theme } = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const bounds = xMin != null && xMax != null ? { xMin, xMax } : null;
  const boundsKey = bounds ? `${bounds.xMin}-${bounds.xMax}` : "none";

  const aligned = useMemo(
    () =>
      toAligned(bounds ? zeroFillGaps(data, bounds.xMin, bounds.xMax) : data),
    // `bounds` is rebuilt every render; `boundsKey` carries its identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data, boundsKey],
  );

  const dataRef = useRef(aligned);
  useEffect(() => {
    dataRef.current = aligned;
  }, [aligned]);

  const seriesKey = JSON.stringify(series);

  const [hoverValues, setHoverValues] = useState<(string | null)[]>(
    series.map(() => null),
  );
  const [hoverTime, setHoverTime] = useState<string | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const colors = resolveTokenColors(el, [
      "line-subtle",
      "muted",
      ...series.flatMap((s) => [s.color, fillToken(s.color)]),
    ]);
    const font = `10px ${resolveTokenValue(el, "--font-mono") || "monospace"}`;
    const axis = {
      grid: { stroke: colors("line-subtle"), width: 1 },
      ticks: { stroke: colors("line-subtle"), width: 1 },
      stroke: colors("muted"),
      font,
    };

    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height,
      series: [
        {},
        ...series.map((s) => ({
          label: s.label,
          stroke: colors(s.color),
          width: s.width ?? 1.5,
          fill: s.fill ? colors(fillToken(s.color)) : undefined,
          points: { show: true, size: 3 },
          spanGaps: s.spanGaps ?? true,
        })),
      ],
      scales: {
        x: {
          time: true,
          auto: !bounds,
          ...(bounds ? { min: bounds.xMin, max: bounds.xMax } : {}),
        },
        y: {
          auto: true,
          range: (_u, min, max) => (min === max ? [0, max + 1] : [0, max]),
        },
      },
      axes: [
        // Tall enough for uPlot's two-line labels at day boundaries.
        { ...axis, size: 44 },
        {
          ...axis,
          size: 40,
          values: (_u, splits) => splits.map((v) => compactTick(v)),
        },
      ],
      legend: { show: false },
      cursor: {
        points: { size: 4 },
        // The x-axis bounds are fixed by the selected time range, so
        // drag-to-zoom is off.
        drag: { setScale: false, x: false, y: false },
      },
      hooks: {
        setCursor: [
          (self: uPlot) => {
            const idx = self.cursor.idx;
            if (idx == null || idx < 0) {
              setHoverValues(series.map(() => null));
              setHoverTime(null);
              return;
            }
            const ts = dataRef.current[0][idx];
            setHoverTime(ts != null ? formatTime(ts) : null);
            setHoverValues(
              series.map((s, i) => {
                const v = dataRef.current[i + 1]?.[idx];
                if (v == null) return null;
                return formatValue(v) + (s.unit ? ` ${s.unit}` : "");
              }),
            );
          },
        ],
      },
    };

    const plot = new uPlot(opts, dataRef.current, el);
    plotRef.current = plot;

    return () => {
      plot.destroy();
      plotRef.current = null;
    };
    // The plot is rebuilt only when its shape or the theme changes; data
    // updates go through `setData` below. `seriesKey` stands in for
    // `series`, whose identity changes every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seriesKey, height, boundsKey, theme]);

  useEffect(() => {
    plotRef.current?.setData(aligned);
    if (bounds) {
      plotRef.current?.setScale("x", { min: bounds.xMin, max: bounds.xMax });
    }
    // See `aligned` above for why `boundsKey` stands in for `bounds`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aligned, boundsKey]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      if (plotRef.current && el.clientWidth > 0) {
        plotRef.current.setSize({ width: el.clientWidth, height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [height]);

  return (
    <div>
      <div ref={containerRef} className="w-full" />
      <Legend series={series} values={hoverValues} time={hoverTime} />
    </div>
  );
}

function Legend({
  series,
  values,
  time,
}: {
  readonly series: readonly ChartSeries[];
  readonly values: readonly (string | null)[];
  readonly time: string | null;
}) {
  return (
    <div
      className="mt-1 grid gap-x-2 gap-y-0.5 font-mono text-xs"
      style={{
        gridTemplateColumns: `repeat(${series.length + 1}, minmax(0, 1fr))`,
      }}
    >
      <div className="text-muted">Time</div>
      {series.map((s, i) => (
        <div key={i} className="flex min-w-0 items-center gap-1.5">
          <span
            className={cn(
              "inline-block size-2 shrink-0",
              seriesBgClassName[s.color],
            )}
          />
          <span className="min-w-0 truncate text-muted">{s.label}</span>
        </div>
      ))}
      <div className="truncate text-soft">{time ?? "—"}</div>
      {series.map((_, i) => (
        <div key={i} className="truncate text-soft">
          {values[i] ?? "—"}
        </div>
      ))}
    </div>
  );
}

/// uPlot wants the x column free of nulls; timestamps never are, so a
/// stray null becomes 0 rather than needing a cast.
function toAligned(data: (number | null)[][]): uPlot.AlignedData {
  const [xs = [], ...ys] = data;
  return [xs.map((x) => x ?? 0), ...ys];
}

function formatTime(seconds: number): string {
  return new Date(seconds * 1000).toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatValue(v: number): string {
  if (v === 0) return "0";
  if (v < 1) return v.toFixed(2);
  if (v < 100) return v.toFixed(1);
  return Math.round(v).toLocaleString();
}
