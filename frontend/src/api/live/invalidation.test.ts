import { describe, expect, it } from "vitest";

import type { DaemonEvent } from "../client.ts";
import { eventsAfter } from "./invalidation.ts";

const reload = (at_ms: number): DaemonEvent => ({
  type: "config_reloaded",
  at_ms,
  changed_services: [],
});

describe("eventsAfter", () => {
  it("treats the first history as backfill, not news", () => {
    expect(eventsAfter([reload(1)], null)).toEqual([]);
  });

  it("returns everything once an empty history has been seen", () => {
    expect(eventsAfter([reload(1)], "")).toEqual([reload(1)]);
  });

  it("returns the events after the newest one seen", () => {
    const seen = JSON.stringify(reload(2));
    expect(eventsAfter([reload(1), reload(2), reload(3)], seen)).toEqual([
      reload(3),
    ]);
    expect(eventsAfter([reload(1), reload(2)], seen)).toEqual([]);
  });
});
