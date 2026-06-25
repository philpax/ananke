// uPlot wrapper that handles creation, data updates, and resize
// without recreating the chart. Reads theme colours from CSS
// variables so it adapts to dark/light mode. Renders its own legend
// below the chart as a two-row grid: series labels on top, values on
// hover.

import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

function readVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  return (
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() ||
    fallback
  );
}

export function Chart({ data, series, height = 160, xMin, xMax }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const dataRef = useRef(data);
  useEffect(() => {
    dataRef.current = data;
  }, [data]);
  const seriesKey = JSON.stringify(series);
  const hasBounds = xMin != null && xMax != null;
  const boundsKey = hasBounds ? `${xMin}-${xMax}` : "none";

  const [hoverValues, setHoverValues] = useState<(string | null)[]>(
    series.map(() => null),
  );
  const [hoverTime, setHoverTime] = useState<string | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const border = readVar("--color-border-default", "#232b3a");
    const tertiary = readVar("--color-text-tertiary", "#5b6477");

    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height,
      series: [
        {},
        ...series.map((s) => ({
          label: s.label,
          stroke: s.stroke,
          width: s.width ?? 1.5,
          fill: s.fill,
          points: { show: true, size: 3 },
        })),
      ],
      scales: {
        x: {
          time: true,
          auto: !hasBounds,
          ...(hasBounds ? { min: xMin!, max: xMax! } : {}),
        },
        y: { auto: true },
      },
      axes: [
        {
          grid: { stroke: border, width: 1 },
          ticks: { stroke: border, width: 1 },
          stroke: tertiary,
          font: "10px 'IBM Plex Mono'",
          size: 32,
        },
        {
          grid: { stroke: border, width: 1 },
          ticks: { stroke: border, width: 1 },
          stroke: tertiary,
          font: "10px 'IBM Plex Mono'",
          size: 36,
        },
      ],
      legend: {
        show: false,
      },
      cursor: {
        points: { size: 4 },
        // Disable drag-to-zoom — the x-axis bounds are fixed by the
        // selected time range and should not be user-adjustable.
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
            const ts = dataRef.current[0]?.[idx];
            setHoverTime(
              ts != null
                ? new Date(ts * 1000).toLocaleTimeString(undefined, {
                    hour12: false,
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })
                : null,
            );
            const vals = series.map((s, i) => {
              const v = dataRef.current[i + 1]?.[idx];
              if (v == null) return null;
              return formatValue(v) + (s.unit ? ` ${s.unit}` : "");
            });
            setHoverValues(vals);
          },
        ],
      },
    };

    const plot = new uPlot(opts, data as uPlot.AlignedData, el);
    plotRef.current = plot;

    return () => {
      plot.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seriesKey, height, boundsKey]);

  useEffect(() => {
    plotRef.current?.setData(data as uPlot.AlignedData);
    if (hasBounds) {
      plotRef.current?.setScale("x", { min: xMin!, max: xMax! });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, boundsKey]);

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
      <div ref={containerRef} style={{ width: "100%" }} />
      <Legend series={series} values={hoverValues} time={hoverTime} />
    </div>
  );
}

function Legend({
  series,
  values,
  time,
}: {
  series: ChartSeries[];
  values: (string | null)[];
  time: string | null;
}) {
  const cols = series.length + 1;
  return (
    <div
      className="mt-1 grid gap-x-2 gap-y-0.5 text-xs"
      style={{
        gridTemplateColumns: `repeat(${cols}, 1fr)`,
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {/* Row 1: headings */}
      <div className="font-mono text-tertiary">Time</div>
      {series.map((s, i) => (
        <div key={i} className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 shrink-0 rounded-sm"
            style={{ backgroundColor: s.stroke }}
          />
          <span className="truncate font-mono text-tertiary">{s.label}</span>
        </div>
      ))}
      {/* Row 2: values */}
      <div className="font-mono text-secondary">{time ?? "—"}</div>
      {series.map((_, i) => (
        <div key={i} className="font-mono text-secondary">
          {values[i] ?? "—"}
        </div>
      ))}
    </div>
  );
}

function formatValue(v: number): string {
  if (v === 0) return "0";
  if (v < 1) return v.toFixed(2);
  if (v < 100) return v.toFixed(1);
  return Math.round(v).toLocaleString();
}

export type ChartSeries = {
  label: string;
  stroke: string;
  fill?: string;
  width?: number;
  unit?: string;
};

export type ChartProps = {
  data: (number | null)[][];
  series: ChartSeries[];
  height?: number;
  xMin?: number;
  xMax?: number;
};
