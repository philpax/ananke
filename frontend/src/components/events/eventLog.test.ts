import { describe, expect, it } from "vitest";

import type { DaemonEvent, ServiceRestartEntry } from "../../api/client.ts";
import {
  eventTone,
  eventsInRange,
  mergeRestarts,
  toggleType,
  type EventType,
} from "./eventLog.ts";

const started: DaemonEvent = {
  type: "state_changed",
  service: "a",
  from: "starting",
  to: "running",
  at_ms: 20,
};
const reloaded: DaemonEvent = {
  type: "config_reloaded",
  changed_services: ["a"],
  at_ms: 30,
};
const restartedLive: DaemonEvent = {
  type: "auto_restarted",
  service: "a",
  trigger: "error_rate",
  detail: "6 of 10 failed",
  at_ms: 40,
};

function restart(
  service: string,
  at_ms: number,
  trigger = "error_rate",
): ServiceRestartEntry {
  return { service, at_ms, trigger, detail: "stored" };
}

const ALL: ReadonlySet<EventType> = new Set();

describe("mergeRestarts", () => {
  it("adds stored restarts the live history does not hold", () => {
    const merged = mergeRestarts([started], [restart("a", 5)]);
    expect(merged).toHaveLength(2);
    expect(merged[1]).toMatchObject({ type: "auto_restarted", at_ms: 5 });
  });

  it("keeps one copy of a restart both sources hold", () => {
    const merged = mergeRestarts([restartedLive], [restart("a", 40)]);
    expect(merged).toEqual([restartedLive]);
  });

  it("tells apart restarts of different services at the same instant", () => {
    const merged = mergeRestarts([restartedLive], [restart("b", 40)]);
    expect(merged).toHaveLength(2);
  });
});

describe("eventsInRange", () => {
  const history = [started, reloaded, restartedLive];

  it("orders newest first", () => {
    const rows = eventsInRange(history, [restart("b", 25)], 0, 100, ALL);
    expect(rows.map((e) => e.at_ms)).toEqual([40, 30, 25, 20]);
  });

  it("keeps only the range, inclusive at both ends", () => {
    const rows = eventsInRange(history, [restart("b", 5)], 20, 30, ALL);
    expect(rows.map((e) => e.at_ms)).toEqual([30, 20]);
  });

  it("filters to every selected type", () => {
    const types = new Set<EventType>(["state_changed", "auto_restarted"]);
    const rows = eventsInRange(history, [restart("b", 25)], 0, 100, types);
    expect(rows.map((e) => e.type)).toEqual([
      "auto_restarted",
      "auto_restarted",
      "state_changed",
    ]);
  });

  it("shows every type when none is selected", () => {
    expect(eventsInRange(history, [], 0, 100, ALL)).toHaveLength(3);
  });
});

describe("toggleType", () => {
  it("adds and removes without touching the original", () => {
    const one = new Set<EventType>(["state_changed"]);
    const two = toggleType(one, "config_reloaded");
    expect([...two]).toEqual(["state_changed", "config_reloaded"]);
    expect([...toggleType(two, "state_changed")]).toEqual(["config_reloaded"]);
    expect([...one]).toEqual(["state_changed"]);
  });
});

describe("eventTone", () => {
  it("takes a state change's tone from the state it entered", () => {
    expect(eventTone(started)).toBe("ok");
    expect(eventTone({ ...started, to: "failed" })).toBe("danger");
    expect(
      eventTone({
        ...started,
        to: "disabled",
        disabled_reason: "user_disabled",
      }),
    ).toBe("neutral");
  });

  it("marks what the daemon corrected on its own", () => {
    expect(eventTone(restartedLive)).toBe("warn");
    expect(eventTone(reloaded)).toBe("neutral");
  });
});
