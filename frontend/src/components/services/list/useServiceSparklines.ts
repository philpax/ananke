// Request-rate sparklines for every service from one metrics query, over
// a rolling hour. The window's start steps once a minute, so the query key
// changes once a minute rather than freezing where it was first opened.

import { useMetrics } from "../../../api/hooks.ts";
import { useSteppedNow } from "../../ui/form/useTimeRange.ts";
import {
  buildServiceSparkline,
  type ServiceSparkData,
} from "./sparklineSeries.ts";

export type ServiceSparklines = {
  readonly byService: ServiceSparkData;
  /// The window's bounds in seconds, as the sparkline's x axis takes them.
  readonly xMin: number;
  readonly xMax: number;
};

export function useServiceSparklines(): ServiceSparklines {
  const now = useSteppedNow(STEP_MS);
  const since = now - WINDOW_MS;
  const metrics = useMetrics({ since, bucket: "1m" });
  return {
    byService: buildServiceSparkline(metrics.data?.buckets ?? []),
    xMin: since / 1000,
    xMax: now / 1000,
  };
}

const STEP_MS = 60_000;
const WINDOW_MS = 60 * 60_000;
