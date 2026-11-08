// The daemon's uptime, kept current between info snapshots. The snapshot is
// only resent on a config reload or a reconnect, so the time since it
// arrived is added on, by this tab's clock alone: no comparison between the
// browser's clock and the daemon's is involved.

import { useEffect, useState } from "react";

/// `uptimeMs` as of `receivedAt` (a `Date.now()` stamp), advanced to now.
export function useUptime(
  uptimeMs: number | undefined,
  receivedAt: number,
): number | undefined {
  const [now, setNow] = useState(() => Date.now());
  const uptime =
    uptimeMs === undefined
      ? undefined
      : uptimeMs + Math.max(0, now - receivedAt);
  const young = uptime !== undefined && uptime < MINUTE_MS;
  useEffect(() => {
    // Seconds show only in the first minute, so tick faster then.
    const id = setInterval(() => setNow(Date.now()), young ? 1_000 : 30_000);
    return () => clearInterval(id);
  }, [young]);
  return uptime;
}

const MINUTE_MS = 60_000;
