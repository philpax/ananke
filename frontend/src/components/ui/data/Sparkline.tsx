// Minimal SVG sparkline for inline activity indicators. No axes, no
// legend — just the line and an optional area fill. Zero is always at
// the bottom so flat-zero lines sit at the base and activity rises from
// there. Colours are token classes, so it follows the theme with no
// resolution step.

import { useMemo } from "react";

import { zeroFillGaps } from "../../../util.ts";
import { cn } from "../cn.ts";
import {
  seriesFillClassName,
  seriesStrokeClassName,
  type SeriesColor,
} from "../chart/series.ts";

type SparklineProps = {
  readonly data: (number | null)[][];
  readonly color: SeriesColor;
  readonly fill?: boolean;
  readonly height?: number;
  readonly xMin?: number;
  readonly xMax?: number;
};

const VIEW_W = 100;

export function Sparkline({
  data,
  color,
  fill = false,
  height = 24,
  xMin,
  xMax,
}: SparklineProps) {
  const paths = useMemo(() => {
    const filled =
      xMin != null && xMax != null ? zeroFillGaps(data, xMin, xMax) : data;
    const ts = filled[0] ?? [];
    const vals = filled[1] ?? [];
    const first = ts[0];
    const last = ts[ts.length - 1];
    if (first == null || last == null) return null;

    const lo = xMin ?? first;
    const span = (xMax ?? last) - lo || 1;
    const yMax = Math.max(...vals.map((v) => v ?? 0), 1);
    const pad = 1;
    const usableH = height - 2 * pad;

    const pts = ts.map((t, i) => {
      const x = (((t ?? lo) - lo) / span) * VIEW_W;
      const y = height - pad - ((vals[i] ?? 0) / yMax) * usableH;
      return `${x},${y}`;
    });
    const line = pts.join(" ");
    const lastX = (((last ?? lo) - lo) / span) * VIEW_W;
    return { line, area: `0,${height} ${line} ${lastX},${height}` };
  }, [data, xMin, xMax, height]);

  const strokeClass = seriesStrokeClassName[color];

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${height}`}
      preserveAspectRatio="none"
      width="100%"
      height={height}
      aria-hidden="true"
    >
      {paths === null ? (
        <line
          x1={0}
          y1={height - 1}
          x2={VIEW_W}
          y2={height - 1}
          className={cn(strokeClass, "opacity-30")}
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      ) : (
        <>
          {fill && (
            <polygon
              points={paths.area}
              className={seriesFillClassName[color]}
            />
          )}
          <polyline
            points={paths.line}
            fill="none"
            className={strokeClass}
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </>
      )}
    </svg>
  );
}
