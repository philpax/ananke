// The pure half of the logs viewer's window: its bounds, and how the seed
// page, older pages loaded by scrolling back, and the live follow combine
// into one list without a hole in the middle.
//
// The window is a contiguous run of the log ending at the newest line. When
// the follow drops its oldest lines, anything held below them would sit on
// the far side of a hole, so it is set aside; scrolling back fetches the
// dropped lines again over HTTP.

import type { LogLine, LogPosition } from "../client.ts";
import { comparePosition, positionOf } from "../live/logFollow.ts";

export type LogWindow =
  | { kind: "relative"; durationMs: number }
  | { kind: "absolute"; sinceMs: number; untilMs: number | null };

export type LogBounds = {
  readonly sinceMs: number;
  /// `null` for a window that runs to now and follows new lines.
  readonly untilMs: number | null;
};

/// Pages fetched by scrolling back, contiguous below `upper`.
export type OlderPages = {
  /// The window the pages belong to; see `windowKey`.
  readonly key: string;
  readonly lines: readonly LogLine[];
  /// The oldest line shown when the first page was fetched.
  readonly upper: LogPosition;
  /// The timestamp bound every page was fetched under.
  readonly untilMs: number;
  /// The cursor for the next page, or null once history is exhausted.
  readonly cursor: string | null;
};

export function resolveBounds(window: LogWindow, nowMs: number): LogBounds {
  if (window.kind === "relative") {
    return { sinceMs: nowMs - window.durationMs, untilMs: null };
  }
  return { sinceMs: window.sinceMs, untilMs: window.untilMs };
}

export function windowKey(name: string | null, bounds: LogBounds): string {
  return `${name ?? ""}|${bounds.sinceMs}|${bounds.untilMs ?? "live"}`;
}

export function compareLogLines(a: LogLine, b: LogLine): number {
  if (a.timestamp_ms !== b.timestamp_ms) return a.timestamp_ms - b.timestamp_ms;
  return comparePosition(positionOf(a), positionOf(b));
}

/// Whether older pages still join the follow: they do until the follow
/// drops lines at or above the pages' top.
export function olderIsAttached(
  older: OlderPages,
  droppedThrough: LogPosition | null,
): boolean {
  return (
    droppedThrough === null || comparePosition(droppedThrough, older.upper) < 0
  );
}

export type WindowParts = {
  readonly seed: readonly LogLine[];
  /// Older pages, already checked to be attached; null if none are.
  readonly older: readonly LogLine[] | null;
  readonly followed: readonly LogLine[];
  /// The newest line the follow dropped, or null if it dropped none.
  readonly droppedThrough: LogPosition | null;
};

/// The window's lines, oldest first, each once.
export function mergeLogWindow(
  parts: WindowParts,
  bounds: LogBounds,
): LogLine[] {
  const map = new Map<string, LogLine>();
  const accept = (l: LogLine) => {
    if (l.timestamp_ms < bounds.sinceMs) return;
    if (bounds.untilMs !== null && l.timestamp_ms > bounds.untilMs) return;
    map.set(`${l.run_id}:${l.seq}`, l);
  };
  // The follow starts right after the seed, so any drop cuts the seed off.
  if (parts.droppedThrough === null) parts.seed.forEach(accept);
  parts.older?.forEach(accept);
  parts.followed.forEach(accept);
  return Array.from(map.values()).sort(compareLogLines);
}

/// The oldest position among `lines`, which is where scrolling back resumes.
export function oldestPosition(
  lines: readonly LogLine[],
): { position: LogPosition; timestampMs: number } | null {
  let oldest: LogLine | null = null;
  for (const l of lines) {
    if (
      oldest === null ||
      comparePosition(positionOf(l), positionOf(oldest)) < 0
    ) {
      oldest = l;
    }
  }
  return oldest === null
    ? null
    : { position: positionOf(oldest), timestampMs: oldest.timestamp_ms };
}
