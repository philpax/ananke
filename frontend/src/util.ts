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
  return new Date(ms).toLocaleString();
}

export function relativeTime(ms: number): string {
  const diff = Date.now() - ms;
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
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
// activity. When consecutive datapoints are more than the gap threshold
// apart (derived from the data's cadence unless `gapSec` is given),
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
  gapSec?: number,
  epsSec = 1,
): (number | null)[][] {
  if (data.length === 0) return [[], []];

  const ts = data[0] as number[];
  const valueSeries = data.slice(1);

  // Derive the gap threshold from the data's own cadence unless overridden:
  // a "gap" is more than 1.5x the median spacing between points. This keeps a
  // run of evenly-spaced buckets connected at any resolution — the fixed
  // 15-min default used to treat every point in the 24h view (1h buckets, now
  // 30m) as a gap and drop to zero between them, rendering continuous traffic
  // as a series of spikes.
  const threshold = gapSec ?? medianGapThreshold(ts);

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
  if (ts[0]! - xMin > threshold) {
    outTs.push(xMin, ts[0]! - epsSec);
    pushZero();
    pushZero();
  }

  for (let i = 0; i < ts.length; i++) {
    // Gap between consecutive points — drop to 0 and back up.
    if (i > 0 && ts[i]! - ts[i - 1]! > threshold) {
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
  if (xMax - last > threshold) {
    outTs.push(last + epsSec, xMax);
    pushZero();
    pushZero();
  }

  return [outTs, ...outVals];
}

// Gap threshold (seconds) derived from a series' own point spacing: 1.5x the
// median interval between consecutive points, so evenly-spaced buckets never
// register as gaps while genuinely missing spans still do. Falls back to
// 15 minutes when there are too few points to estimate a cadence.
function medianGapThreshold(ts: number[]): number {
  if (ts.length < 2) return 900;
  const diffs: number[] = [];
  for (let i = 1; i < ts.length; i++) diffs.push(ts[i]! - ts[i - 1]!);
  diffs.sort((a, b) => a - b);
  const mid = Math.floor(diffs.length / 2);
  const median =
    diffs.length % 2 === 1 ? diffs[mid]! : (diffs[mid - 1]! + diffs[mid]!) / 2;
  return median * 1.5;
}

// Pick a metrics bucket size for a window span so charts keep a
// readable point density (~30-70 points) at any range, including the
// 5m preset and arbitrary custom ranges. The backend accepts any
// duration string here.
export function bucketFor(spanMs: number): string {
  if (spanMs <= 5 * 60_000) return "10s";
  if (spanMs <= 3_600_000) return "1m";
  if (spanMs <= 6 * 3_600_000) return "5m";
  if (spanMs <= 24 * 3_600_000) return "30m";
  return "2h";
}

// Resolve a TimeWindow into concrete metrics-query parameters. For
// relative windows `now` is captured at call time — memoise on the
// window object so the query key doesn't churn every render.
export function metricsWindow(w: {
  kind: string;
  durationMs?: number;
  sinceMs?: number;
  untilMs?: number | null;
}): {
  since: number;
  until: number | undefined;
  end: number;
  bucket: string;
} {
  const now = Date.now();
  const since =
    w.kind === "relative"
      ? now - (w.durationMs ?? 3_600_000)
      : (w.sinceMs ?? now);
  const until =
    w.kind === "absolute" && w.untilMs != null ? w.untilMs : undefined;
  const end = until ?? now;
  return { since, until, end, bucket: bucketFor(end - since) };
}
