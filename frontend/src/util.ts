// Formatting helpers shared across the dashboard. All user-facing
// formatting goes through here so units, precision, and rounding stay
// consistent.

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${n} B`;
}

export function formatBytesShort(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)}G`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)}M`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)}K`;
  return `${n}`;
}

export function formatParameterCount(n: number): string {
  const fmt = (value: number, suffix: string) =>
    `${value < 100 ? value.toFixed(1) : Math.round(value)}${suffix}`;
  if (n >= 1e12) return fmt(n / 1e12, "T");
  if (n >= 1e9) return fmt(n / 1e9, "B");
  if (n >= 1e6) return fmt(n / 1e6, "M");
  if (n >= 1e3) return fmt(n / 1e3, "K");
  return `${n}`;
}

export function formatDuration(ms: number): string {
  if (ms < 1) return "0ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s}s`;
}

export function formatTimestamp(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatTokenRate(tokensPerSecond: number): string {
  if (tokensPerSecond < 1) return tokensPerSecond.toFixed(2);
  if (tokensPerSecond < 100) return tokensPerSecond.toFixed(1);
  return Math.round(tokensPerSecond).toString();
}

export function serviceProxyUrl(port: number): string {
  return `${window.location.protocol}//${window.location.hostname}:${port}`;
}

// Construct the OpenAI API base URL from the daemon's `openai_listen`
// address (e.g. "0.0.0.0:7070"). Uses the browser's hostname (not
// 0.0.0.0) with the port from the listen address. CORS is enabled on
// the daemon, so cross-origin requests work.
export function openaiBaseUrlFromListen(listen: string | undefined): string {
  const parts = (listen ?? "0.0.0.0:7070").split(":");
  const port = parts[parts.length - 1];
  return `http://${window.location.hostname}:${port}`;
}

// Preprocess time-series data so the line drops to 0 during gaps in
// activity. When consecutive datapoints are more than `gapSec` apart,
// inserts a zero just after the earlier point and another just before
// the later one, creating a sharp drop and rise instead of a long
// interpolation. Also pads the chart edges: if the first or last point
// is far from xMin / xMax, inserts zero-points at the boundary so the
// line starts and ends at 0.
//
// This naturally handles three edge cases without special-casing:
//   - No data: returns a flat 0 line across [xMin, xMax].
//   - One datapoint: returns 0 → value → 0 (a visible spike).
//   - Multiple clusters: each cluster gets sharp 0 edges.
//
// The epsilon (`epsSec`) is a small time delta so the drop/rise reads
// as a near-vertical edge rather than a gradual slope. All value
// series share the same timestamps; existing nulls are preserved.
export function zeroFillGaps(
  data: (number | null)[][],
  xMin: number,
  xMax: number,
  gapSec = 900,
  epsSec = 1,
): (number | null)[][] {
  if (data.length === 0) return [[], []];

  const ts = data[0] as number[];
  const valueSeries = data.slice(1);

  // No timestamps — flat 0 line across the full range.
  if (ts.length === 0) {
    const result: (number | null)[][] = [[xMin, xMax]];
    for (let s = 0; s < valueSeries.length; s++) {
      result.push([0, 0]);
    }
    return result;
  }

  const outTs: number[] = [];
  const outVals: (number | null)[][] = valueSeries.map(() => []);

  const pushZero = () => {
    for (let s = 0; s < outVals.length; s++) outVals[s].push(0);
  };

  const pushPoint = (i: number) => {
    for (let s = 0; s < valueSeries.length; s++) {
      outVals[s].push(valueSeries[s][i] ?? null);
    }
  };

  // Leading edge: pad from xMin to just before the first point.
  if (ts[0]! - xMin > gapSec) {
    outTs.push(xMin, ts[0]! - epsSec);
    pushZero();
    pushZero();
  }

  for (let i = 0; i < ts.length; i++) {
    // Gap between consecutive points — drop to 0 and back up.
    if (i > 0 && ts[i]! - ts[i - 1]! > gapSec) {
      outTs.push(ts[i - 1]! + epsSec);
      pushZero();
      outTs.push(ts[i]! - epsSec);
      pushZero();
    }
    outTs.push(ts[i]!);
    pushPoint(i);
  }

  // Trailing edge: pad from just after the last point to xMax.
  const last = ts[ts.length - 1]!;
  if (xMax - last > gapSec) {
    outTs.push(last + epsSec, xMax);
    pushZero();
    pushZero();
  }

  return [outTs, ...outVals];
}

export type TimeRange = { label: string; ms: number; bucket: string };

export const RANGES: TimeRange[] = [
  { label: "1h", ms: 3_600_000, bucket: "1m" },
  { label: "6h", ms: 6 * 3_600_000, bucket: "5m" },
  { label: "24h", ms: 24 * 3_600_000, bucket: "1h" },
];
