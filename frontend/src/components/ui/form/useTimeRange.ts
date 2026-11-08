// The URL-held time range as React state, and its bounds as they roll
// forward with the clock.

import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { bucketFor, type BucketSize } from "../../../util.ts";
import {
  parseTimeRange,
  resolveTimeRange,
  withTimeRange,
  type TimeRange,
} from "./timeRange.ts";

/// The current range and a setter that rewrites only the range's
/// parameters. Changing the range replaces the history entry, so Back
/// leaves the page rather than stepping through every range tried.
export function useTimeRange(): readonly [
  TimeRange,
  (range: TimeRange) => void,
] {
  const [params, setParams] = useSearchParams();
  const range = parseTimeRange(params);
  const setRange = (next: TimeRange) =>
    setParams((prev) => withTimeRange(prev, next), { replace: true });
  return [range, setRange];
}

export type ResolvedRange = {
  readonly since: number;
  /// `undefined` for a rolling range, which runs to now.
  readonly until: number | undefined;
  /// The right edge of a chart: `until`, or now.
  readonly end: number;
  readonly bucket: BucketSize;
};

/// A range's metrics-query bounds. A rolling range's "now" advances in
/// `stepMs` steps, floored to the step, so a query keyed on it changes
/// once per step rather than on every render, and a long-lived tab keeps
/// showing the last hour rather than everything since it was opened.
export function useResolvedRange(
  range: TimeRange,
  stepMs = DEFAULT_STEP_MS,
): ResolvedRange {
  const now = useSteppedNow(stepMs, range.kind === "rolling");
  const { sinceMs, untilMs } = resolveTimeRange(range, now);
  const end = untilMs ?? now;
  return {
    since: sinceMs,
    until: untilMs ?? undefined,
    end,
    bucket: bucketFor(end - sinceMs),
  };
}

/// Wall-clock time floored to `stepMs`, re-read every step while `active`.
export function useSteppedNow(stepMs: number, active = true): number {
  const [now, setNow] = useState(() => floorTo(Date.now(), stepMs));
  useEffect(() => {
    if (!active) return;
    const tick = () => setNow(floorTo(Date.now(), stepMs));
    tick();
    const id = setInterval(tick, stepMs);
    return () => clearInterval(id);
  }, [stepMs, active]);
  return now;
}

const DEFAULT_STEP_MS = 30_000;

function floorTo(ms: number, stepMs: number): number {
  return Math.floor(ms / stepMs) * stepMs;
}
