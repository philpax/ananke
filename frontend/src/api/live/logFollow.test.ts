import { describe, expect, it } from "vitest";

import type { LiveFeed, LogLine } from "../client.ts";
import {
  EMPTY_FOLLOW,
  applyFollowMessage,
  comparePosition,
  resumeFeed,
  type FollowMessage,
  type FollowState,
} from "./logFollow.ts";

function line(run_id: number, seq: number): LogLine {
  return {
    timestamp_ms: seq,
    stream: "stdout",
    line: `${run_id}:${seq}`,
    run_id,
    seq,
  };
}

function lines(...ls: LogLine[]): FollowMessage {
  return { type: "log_lines", id: "f0", lines: ls };
}

function apply(
  messages: readonly FollowMessage[],
  cap = 100,
  from: FollowState = EMPTY_FOLLOW,
): FollowState {
  return messages.reduce((s, m) => applyFollowMessage(s, m, cap), from);
}

const seqs = (s: FollowState) => s.lines.map((l) => `${l.run_id}:${l.seq}`);

describe("comparePosition", () => {
  it("orders by run, then seq", () => {
    expect(comparePosition({ run_id: 1, seq: 9 }, { run_id: 2, seq: 1 })).toBe(
      -1,
    );
    expect(comparePosition({ run_id: 2, seq: 3 }, { run_id: 2, seq: 1 })).toBe(
      1,
    );
    expect(comparePosition({ run_id: 2, seq: 3 }, { run_id: 2, seq: 3 })).toBe(
      0,
    );
  });
});

describe("applyFollowMessage", () => {
  it("appends lines and tracks the newest", () => {
    const s = apply([lines(line(1, 1), line(1, 2)), lines(line(2, 1))]);
    expect(seqs(s)).toEqual(["1:1", "1:2", "2:1"]);
    expect(s.last).toEqual({ run_id: 2, seq: 1 });
  });

  it("ignores lines at or below the newest held", () => {
    const s = apply([
      lines(line(1, 1), line(1, 2)),
      lines(line(1, 2), line(1, 3)),
      lines(line(1, 1)),
    ]);
    expect(seqs(s)).toEqual(["1:1", "1:2", "1:3"]);
  });

  it("sums gaps and marks catching up", () => {
    const gap = (skipped: number): FollowMessage => ({
      type: "log_gap",
      id: "f0",
      after: null,
      before: { run_id: 1, seq: 1 },
      skipped,
    });
    const s = apply([gap(5), gap(7)]);
    expect(s.skipped).toBe(12);
    expect(s.caughtUp).toBe(false);
    expect(apply([{ type: "log_caught_up", id: "f0" }], 100, s).caughtUp).toBe(
      true,
    );
  });

  it("drops the oldest lines past the cap and keeps the newest", () => {
    const s = apply(
      [
        lines(line(1, 1), line(1, 2), line(1, 3)),
        lines(line(1, 4), line(1, 5)),
      ],
      3,
    );
    expect(seqs(s)).toEqual(["1:3", "1:4", "1:5"]);
    expect(s.droppedThrough).toEqual({ run_id: 1, seq: 2 });
    expect(s.last).toEqual({ run_id: 1, seq: 5 });
  });

  it("records errors until lines arrive", () => {
    const error: FollowMessage = {
      type: "error",
      id: "f0",
      status: 404,
      error: {
        error: {
          code: "service_not_found",
          message: "service `ghost` not found",
          type: "invalid_request_error",
        },
      },
    };
    const failed = apply([error]);
    expect(failed.error?.error.code).toBe("service_not_found");
    expect(apply([lines(line(1, 1))], 100, failed).error).toBeNull();
  });
});

describe("resumeFeed", () => {
  const feed: LiveFeed = {
    kind: "logs",
    service: "demo",
    after: { run_id: 1, seq: 4 },
  };

  it("keeps the original position until a line arrives", () => {
    expect(resumeFeed(feed, EMPTY_FOLLOW)).toEqual(feed);
  });

  it("prefers the newest line held", () => {
    const s = apply([lines(line(2, 7))]);
    expect(resumeFeed(feed, s).after).toEqual({ run_id: 2, seq: 7 });
  });
});
