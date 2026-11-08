import { describe, expect, it } from "vitest";

import type { DisableReason, ServiceStateKind } from "../../../api/client.ts";
import type { Tone } from "../tone.ts";
import {
  needsAttention,
  reasonLabelKey,
  stateLabelKey,
  stateTone,
} from "./stateTone.ts";

describe("stateTone", () => {
  it.each<[ServiceStateKind, Tone]>([
    ["running", "ok"],
    ["starting", "warn"],
    ["draining", "warn"],
    ["failed", "danger"],
    ["idle", "neutral"],
    ["stopped", "neutral"],
    ["evicted", "neutral"],
    ["unknown", "neutral"],
    ["disabled", "neutral"],
  ])("%s is %s", (state, tone) => {
    expect(stateTone(state)).toBe(tone);
  });

  it.each<[DisableReason, Tone]>([
    ["user_disabled", "neutral"],
    ["launch_failed", "danger"],
    ["health_timeout", "danger"],
    ["oom", "danger"],
    ["auto_restart_loop", "danger"],
  ])("disabled for %s is %s", (reason, tone) => {
    expect(stateTone("disabled", reason)).toBe(tone);
  });
});

describe("needsAttention", () => {
  it("counts failed and fault-disabled services", () => {
    expect(needsAttention({ state: "failed" })).toBe(true);
    expect(
      needsAttention({ state: "disabled", disabled_reason: "health_timeout" }),
    ).toBe(true);
    expect(
      needsAttention({ state: "disabled", disabled_reason: "user_disabled" }),
    ).toBe(false);
    expect(needsAttention({ state: "running" })).toBe(false);
  });
});

describe("label keys", () => {
  it("name the state and the reason", () => {
    expect(stateLabelKey("running")).toBe("services.states.running");
    expect(reasonLabelKey("oom")).toBe("services.disabledReasons.oom");
  });
});
