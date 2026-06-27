// Minimal SVG sparkline for inline activity indicators. No axes, no
// legend — just the line (and optional area fill). Zero is always at
// the bottom so flat-zero lines sit at the base and activity rises
// from there. Used for service-row activity sparklines.

import { useMemo } from "react";

import { zeroFillGaps } from "../../util.ts";

type SparklineProps = {
  data: (number | null)[][];
  color: string;
  height?: number;
  fill?: string;
  xMin?: number;
  xMax?: number;
};

const VIEW_W = 100;

export function Sparkline({
  data,
  color,
  height = 24,
  fill,
  xMin,
  xMax,
}: SparklineProps) {
  const { linePath, fillPath, flatLine } = useMemo(() => {
    const hasBounds = xMin != null && xMax != null;
    const filled = hasBounds ? zeroFillGaps(data, xMin!, xMax!) : data;
    const ts = (filled[0] ?? []) as number[];
    const vals = (filled[1] ?? []) as (number | null)[];

    if (ts.length === 0) {
      return { linePath: "", fillPath: "", flatLine: true };
    }

    const lo = xMin ?? ts[0]!;
    const hi = xMax ?? ts[ts.length - 1]!;
    const span = hi - lo || 1;

    const yMax = Math.max(...vals.map((v) => v ?? 0), 1);
    const pad = 1;
    const usableH = height - 2 * pad;

    const pts = ts.map((t, i) => {
      const x = ((t - lo) / span) * VIEW_W;
      const y = height - pad - ((vals[i] ?? 0) / yMax) * usableH;
      return [x, y] as const;
    });

    const line = pts.map(([x, y]) => `${x},${y}`).join(" ");
    const area = `0,${height} ${line} ${pts[pts.length - 1]![0]},${height}`;

    return { linePath: line, fillPath: area, flatLine: false };
  }, [data, xMin, xMax, height]);
  if (flatLine) {
    return (
      <svg
        viewBox={`0 0 ${VIEW_W} ${height}`}
        preserveAspectRatio="none"
        width="100%"
        height={height}
      >
        <line
          x1={0}
          y1={height - 1}
          x2={VIEW_W}
          y2={height - 1}
          stroke={color}
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
          opacity={0.3}
        />
      </svg>
    );
  }

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${height}`}
      preserveAspectRatio="none"
      width="100%"
      height={height}
    >
      {fill && <polygon points={fillPath} fill={fill} />}
      <polyline
        points={linePath}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
