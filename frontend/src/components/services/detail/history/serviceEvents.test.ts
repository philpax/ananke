import { describe, expect, it } from "vitest";

import type { DaemonEvent } from "../../../../api/client.ts";
import { serviceEvents } from "./serviceEvents.ts";

const history: DaemonEvent[] = [
  {
    type: "state_changed",
    service: "a",
    from: "idle",
    to: "starting",
    at_ms: 10,
  },
  {
    type: "state_changed",
    service: "b",
    from: "idle",
    to: "starting",
    at_ms: 11,
  },
  { type: "allocation_changed", service: "a", reservations: {}, at_ms: 12 },
  { type: "config_reloaded", changed_services: ["a"], at_ms: 13 },
  { type: "config_reloaded", changed_services: ["b"], at_ms: 14 },
  {
    type: "auto_restarted",
    service: "a",
    trigger: "error_rate",
    detail: "",
    at_ms: 20,
  },
  {
    type: "state_changed",
    service: "a",
    from: "idle",
    to: "starting",
    at_ms: 500,
  },
];

describe("serviceEvents", () => {
  it("keeps the service's events in range, newest first", () => {
    const events = serviceEvents(history, [], "a", 0, 100);
    expect(events.map((e) => [e.type, e.at_ms])).toEqual([
      ["auto_restarted", 20],
      ["config_reloaded", 13],
      ["state_changed", 10],
    ]);
  });

  it("merges persisted restarts and drops ones the history holds", () => {
    const events = serviceEvents(
      history,
      [
        { service: "a", trigger: "error_rate", detail: "dup", at_ms: 20 },
        { service: "a", trigger: "periodic", detail: "old", at_ms: 5 },
        { service: "b", trigger: "periodic", detail: "", at_ms: 6 },
        { service: "a", trigger: "periodic", detail: "", at_ms: 1_000 },
      ],
      "a",
      0,
      100,
    );
    expect(events.filter((e) => e.type === "auto_restarted")).toEqual([
      expect.objectContaining({ at_ms: 20, detail: "" }),
      expect.objectContaining({ at_ms: 5, detail: "old" }),
    ]);
  });
});
