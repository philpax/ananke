import { describe, expect, it } from "vitest";

import type { LogLine } from "../client.ts";
import {
  mergeLogWindow,
  oldestPosition,
  olderIsAttached,
  resolveBounds,
  type OlderPages,
} from "./logWindow.ts";

function line(run_id: number, seq: number, timestamp_ms = seq): LogLine {
  return {
    timestamp_ms,
    stream: "stdout",
    line: `${run_id}:${seq}`,
    run_id,
    seq,
  };
}

const keys = (ls: readonly LogLine[]) => ls.map((l) => `${l.run_id}:${l.seq}`);
const LIVE = { sinceMs: 0, untilMs: null };

describe("mergeLogWindow", () => {
  it("dedupes the seed against the follow and sorts", () => {
    const merged = mergeLogWindow(
      {
        seed: [line(1, 2), line(1, 1)],
        older: null,
        followed: [line(1, 2), line(1, 3)],
        droppedThrough: null,
      },
      LIVE,
    );
    expect(keys(merged)).toEqual(["1:1", "1:2", "1:3"]);
  });

  it("respects the window's bounds", () => {
    const merged = mergeLogWindow(
      {
        seed: [line(1, 1, 5), line(1, 2, 15), line(1, 3, 25)],
        older: null,
        followed: [],
        droppedThrough: null,
      },
      { sinceMs: 10, untilMs: 20 },
    );
    expect(keys(merged)).toEqual(["1:2"]);
  });

  it("sets the seed aside once the follow drops lines", () => {
    const merged = mergeLogWindow(
      {
        seed: [line(1, 1)],
        older: null,
        followed: [line(1, 5), line(1, 6)],
        droppedThrough: { run_id: 1, seq: 4 },
      },
      LIVE,
    );
    expect(keys(merged)).toEqual(["1:5", "1:6"]);
  });

  it("includes attached older pages", () => {
    const merged = mergeLogWindow(
      {
        seed: [],
        older: [line(1, 3), line(1, 4)],
        followed: [line(1, 5)],
        droppedThrough: { run_id: 1, seq: 4 },
      },
      LIVE,
    );
    expect(keys(merged)).toEqual(["1:3", "1:4", "1:5"]);
  });
});

describe("olderIsAttached", () => {
  const older: OlderPages = {
    key: "demo",
    lines: [],
    upper: { run_id: 1, seq: 5 },
    untilMs: 5,
    cursor: null,
  };

  it("holds until the follow drops the pages' top", () => {
    expect(olderIsAttached(older, null)).toBe(true);
    expect(olderIsAttached(older, { run_id: 1, seq: 4 })).toBe(true);
    expect(olderIsAttached(older, { run_id: 1, seq: 5 })).toBe(false);
    expect(olderIsAttached(older, { run_id: 2, seq: 1 })).toBe(false);
  });
});

describe("oldestPosition", () => {
  it("finds the oldest by position, not timestamp", () => {
    expect(oldestPosition([line(2, 1, 1), line(1, 9, 50)])).toEqual({
      position: { run_id: 1, seq: 9 },
      timestampMs: 50,
    });
    expect(oldestPosition([])).toBeNull();
  });
});

describe("resolveBounds", () => {
  it("runs a relative window to now", () => {
    expect(resolveBounds({ kind: "relative", durationMs: 100 }, 1_000)).toEqual(
      { sinceMs: 900, untilMs: null },
    );
  });
});
