// The logs viewer's data: a seed of history over HTTP, then, for a window
// that runs to now, a follow on the live socket from where the seed ended.
// Scrolling back past what is held fetches older pages over HTTP.

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api, type LogLine, type LogPosition } from "../client.ts";
import { MAX_FOLLOWED_LINES, MAX_HELD_LINES } from "../live/logFollow.ts";
import { useLiveStatus } from "../live/useLive.ts";
import { useLogFollow } from "../live/useLogFollow.ts";
import {
  mergeLogWindow,
  oldestPosition,
  olderIsAttached,
  resolveBounds,
  windowKey,
  type LogBounds,
  type LogWindow,
  type OlderPages,
} from "./logWindow.ts";

const LOGS_PAGE_SIZE = 1_000;

export type LogWindowResult = {
  readonly lines: readonly LogLine[];
  readonly isLive: boolean;
  /// Lines the follow reported skipping because too many arrived at once.
  readonly skipped: number;
  /// Older lines in the window are not held; scrolling back fetches them.
  readonly truncated: boolean;
  readonly hasOlder: boolean;
  readonly loadingOlder: boolean;
  readonly loadOlder: () => void;
  readonly error: Error | null;
  readonly loading: boolean;
};

export function useLogWindow(
  name: string | null,
  window: LogWindow,
): LogWindowResult {
  // The window's start is fixed when it is chosen, not on every render, and
  // a window is the same window when its fields are, whatever its identity.
  const identity = windowIdentity(window);
  const [held, setHeld] = useState(() => ({
    identity,
    bounds: resolveBounds(window, nowMs()),
  }));
  if (held.identity !== identity) {
    setHeld({ identity, bounds: resolveBounds(window, nowMs()) });
  }
  const bounds =
    held.identity === identity ? held.bounds : resolveBounds(window, nowMs());
  const isLiveWindow = bounds.untilMs === null;
  const liveStatus = useLiveStatus();
  const key = windowKey(name, bounds);

  const seed = useQuery({
    queryKey: ["log-seed", name, bounds.sinceMs, bounds.untilMs],
    queryFn: () => fetchSeedWindow(name ?? "", bounds),
    enabled: name !== null,
    // A live window's follow keeps it current from where the seed ended.
    staleTime: isLiveWindow ? Infinity : 0,
  });

  const [older, setOlder] = useState<OlderPages | null>(null);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [olderError, setOlderError] = useState<Error | null>(null);
  const heldOlder = older?.key === key ? older : null;

  const follow = useLogFollow(
    isLiveWindow ? name : null,
    isLiveWindow ? seed.data?.followAfter : undefined,
    heldOlder ? MAX_HELD_LINES : MAX_FOLLOWED_LINES,
  );
  const attachedOlder =
    heldOlder && olderIsAttached(heldOlder, follow.droppedThrough)
      ? heldOlder
      : null;

  const lines = useMemo(
    () =>
      mergeLogWindow(
        {
          seed: seed.data?.lines ?? [],
          older: attachedOlder?.lines ?? null,
          followed: follow.lines,
          droppedThrough: follow.droppedThrough,
        },
        bounds,
      ),
    [seed.data, attachedOlder, follow.lines, follow.droppedThrough, bounds],
  );

  const truncated = attachedOlder
    ? attachedOlder.cursor !== null
    : (seed.data?.truncated ?? false) || follow.droppedThrough !== null;

  const loadOlder = () => {
    if (name === null || loadingOlder || !truncated) return;
    const start = attachedOlder ?? freshOlder(key, lines);
    if (start === null) return;
    setLoadingOlder(true);
    api
      .getLogs(name, {
        since: bounds.sinceMs,
        until: start.untilMs,
        limit: LOGS_PAGE_SIZE,
        before: start.cursor ?? undefined,
      })
      .then(
        (resp) => {
          setOlder({
            ...start,
            lines: [...start.lines, ...resp.logs],
            cursor: resp.next_cursor ?? null,
          });
          setOlderError(null);
        },
        (e: unknown) =>
          setOlderError(e instanceof Error ? e : new Error(String(e))),
      )
      .finally(() => setLoadingOlder(false));
  };

  const followError = follow.error
    ? new Error(follow.error.error.message)
    : null;
  return {
    lines,
    isLive: isLiveWindow && liveStatus === "open" && follow.caughtUp,
    skipped: follow.skipped,
    truncated,
    hasOlder: truncated,
    loadingOlder,
    loadOlder,
    error: seed.error ?? followError ?? olderError,
    loading: seed.isPending && name !== null,
  };
}

function windowIdentity(window: LogWindow): string {
  return window.kind === "relative"
    ? `relative:${window.durationMs}`
    : `absolute:${window.sinceMs}:${window.untilMs ?? ""}`;
}

function nowMs(): number {
  return Date.now();
}

/// A first page of older history, starting below the oldest line shown.
function freshOlder(key: string, lines: readonly LogLine[]): OlderPages | null {
  const oldest = oldestPosition(lines);
  if (oldest === null) return null;
  return {
    key,
    lines: [],
    upper: oldest.position,
    untilMs: oldest.timestampMs,
    cursor: null,
  };
}

type SeedWindow = {
  readonly lines: LogLine[];
  readonly truncated: boolean;
  /// Where a follow continues from; null for an empty log, and undefined
  /// for a window with an end, which does not follow.
  readonly followAfter: LogPosition | null | undefined;
};

/// The seed is the newest page of the window; scrolling back fetches the
/// rest a page at a time, so opening a service never waits on a day of logs.
async function fetchSeedWindow(
  name: string,
  bounds: LogBounds,
): Promise<SeedWindow> {
  const resp = await api.getLogs(name, {
    since: bounds.sinceMs,
    // A live window omits `until`, so the page says where it ends.
    until: bounds.untilMs ?? undefined,
    limit: LOGS_PAGE_SIZE,
  });
  return {
    lines: resp.logs,
    truncated: Boolean(resp.next_cursor),
    followAfter:
      bounds.untilMs === null ? (resp.follow_after ?? null) : undefined,
  };
}
