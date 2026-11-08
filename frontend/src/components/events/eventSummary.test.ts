import i18n from "i18next";
import { describe, expect, it } from "vitest";

import { summarizeEvent } from "./eventSummary.ts";

const t = i18n.t.bind(i18n);

describe("summarizeEvent", () => {
  it("names both states of a transition", () => {
    expect(
      summarizeEvent(
        {
          type: "state_changed",
          service: "qwen",
          from: "idle",
          to: "starting",
          at_ms: 1,
        },
        t,
      ),
    ).toBe("idle → starting");
  });

  it("says why a service was disabled", () => {
    expect(
      summarizeEvent(
        {
          type: "state_changed",
          service: "qwen",
          from: "starting",
          to: "disabled",
          disabled_reason: "health_timeout",
          at_ms: 1,
        },
        t,
      ),
    ).toBe("starting → disabled (health timeout)");
  });

  it("includes an auto-restart's detail when it has one", () => {
    expect(
      summarizeEvent(
        {
          type: "auto_restarted",
          service: "qwen",
          trigger: "error_rate",
          detail: "6 of 10 failed",
          at_ms: 1,
        },
        t,
      ),
    ).toBe("auto-restarted (error_rate): 6 of 10 failed");
  });

  it("lists where an allocation now sits, by device", () => {
    expect(
      summarizeEvent(
        {
          type: "allocation_changed",
          service: "qwen",
          reservations: { "gpu:1": 2 * 1024 ** 3, cpu: 0, "gpu:0": 1024 ** 3 },
          at_ms: 1,
        },
        t,
      ),
    ).toBe("holds 1.00 GiB on gpu:0, 2.00 GiB on gpu:1");
  });

  it("says when an allocation is released", () => {
    expect(
      summarizeEvent(
        {
          type: "allocation_changed",
          service: "qwen",
          reservations: {},
          at_ms: 1,
        },
        t,
      ),
    ).toBe("released its reservations");
  });

  it("names the services a reload changed", () => {
    expect(
      summarizeEvent(
        { type: "config_reloaded", changed_services: ["a", "b"], at_ms: 1 },
        t,
      ),
    ).toBe("configuration reloaded from disk, changing 2 services: a, b");
  });
});
