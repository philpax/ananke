// The pure half of a `logs` follow on the live socket: folding the daemon's
// frames into a buffer of lines, and where to resume after a reconnect.
//
// The daemon never repeats a line within one follow, but a reconnect resumes
// from the last line this buffer holds, so anything at or below it is dropped
// here rather than trusted to be absent.

import type {
  ApiError,
  LiveFeed,
  LiveMessage,
  LogLine,
  LogPosition,
} from "../client.ts";

/// Lines a follow keeps while nothing older is loaded. Past this the oldest
/// are dropped; they stay readable by paging back over HTTP.
export const MAX_FOLLOWED_LINES = 20_000;

/// Lines a follow keeps while older history is loaded under it, so new lines
/// do not immediately cut the loaded pages off from the live tail.
export const MAX_HELD_LINES = 2 * MAX_FOLLOWED_LINES;

/// The frames that belong to a follow.
export type FollowMessage = Extract<
  LiveMessage,
  { type: "log_lines" | "log_gap" | "log_caught_up" | "error" }
>;

export type FollowState = {
  readonly lines: readonly LogLine[];
  /// The newest position received, kept across trims for resuming.
  readonly last: LogPosition | null;
  /// The newest position trimmed off the front, or null if none was.
  readonly droppedThrough: LogPosition | null;
  /// Lines the daemon reported skipping in `log_gap` frames.
  readonly skipped: number;
  /// The daemon has sent everything stored as of its last read.
  readonly caughtUp: boolean;
  readonly error: ApiError | null;
};

export const EMPTY_FOLLOW: FollowState = {
  lines: [],
  last: null,
  droppedThrough: null,
  skipped: 0,
  caughtUp: false,
  error: null,
};

export function comparePosition(a: LogPosition, b: LogPosition): number {
  if (a.run_id !== b.run_id) return a.run_id < b.run_id ? -1 : 1;
  if (a.seq !== b.seq) return a.seq < b.seq ? -1 : 1;
  return 0;
}

export function positionOf(line: LogLine): LogPosition {
  return { run_id: line.run_id, seq: line.seq };
}

/// Fold one frame into the buffer, keeping at most `cap` lines.
export function applyFollowMessage(
  state: FollowState,
  message: FollowMessage,
  cap: number,
): FollowState {
  switch (message.type) {
    case "log_lines": {
      const fresh = message.lines.filter(
        (l) =>
          state.last === null || comparePosition(positionOf(l), state.last) > 0,
      );
      if (fresh.length === 0) return state;
      const lastLine = fresh[fresh.length - 1];
      return trim(
        {
          ...state,
          lines: [...state.lines, ...fresh],
          last: lastLine ? positionOf(lastLine) : state.last,
          error: null,
        },
        cap,
      );
    }
    case "log_gap":
      return { ...state, skipped: state.skipped + message.skipped };
    case "log_caught_up":
      return { ...state, caughtUp: true, error: null };
    case "error":
      return { ...state, error: message.error };
    default: {
      const unreachable: never = message;
      return unreachable;
    }
  }
}

/// Drop the oldest lines past `cap`.
export function trim(state: FollowState, cap: number): FollowState {
  const excess = state.lines.length - cap;
  if (excess <= 0) return state;
  const dropped = state.lines[excess - 1];
  return {
    ...state,
    lines: state.lines.slice(excess),
    droppedThrough: dropped ? positionOf(dropped) : state.droppedThrough,
  };
}

/// The feed to send when (re)starting a follow: after the newest line held,
/// or the original position if nothing has arrived yet.
export function resumeFeed(feed: LiveFeed, state: FollowState): LiveFeed {
  return state.last === null ? feed : { ...feed, after: state.last };
}
