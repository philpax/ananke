import { describe, expect, it } from "vitest";

import type { ServiceStateKind } from "../../../api/client.ts";
import {
  actionOutcome,
  availableActions,
  canChat,
  serviceOpenUrl,
} from "./serviceActions.ts";

describe("availableActions", () => {
  it.each<[ServiceStateKind, "start" | "stop" | null, boolean, string]>([
    ["running", "stop", true, "disable"],
    ["starting", "stop", true, "disable"],
    ["draining", "stop", true, "disable"],
    ["idle", "start", false, "disable"],
    ["stopped", "start", false, "disable"],
    ["failed", "start", false, "disable"],
    ["evicted", "start", false, "disable"],
    ["disabled", null, false, "enable"],
    ["unknown", null, false, "disable"],
  ])("%s", (state, toggle, restart, power) => {
    expect(availableActions(state)).toEqual({ toggle, restart, power });
  });
});

describe("canChat", () => {
  it("needs an enabled chat model on the OpenAI API", () => {
    expect(canChat({ openai_compat: true, state: "idle" })).toBe(true);
    expect(canChat({ openai_compat: true, state: "disabled" })).toBe(false);
    expect(canChat({ openai_compat: false, state: "idle" })).toBe(false);
    expect(
      canChat({ openai_compat: true, modality: "embedding", state: "idle" }),
    ).toBe(false);
  });
});

describe("serviceOpenUrl", () => {
  it("joins the tab's host with the port", () => {
    expect(
      serviceOpenUrl({ protocol: "http:", hostname: "box.lan" }, 8080),
    ).toBe("http://box.lan:8080/");
  });

  it("brackets an IPv6 host", () => {
    expect(serviceOpenUrl({ protocol: "https:", hostname: "::1" }, 80)).toBe(
      "https://[::1]:80/",
    );
    expect(
      serviceOpenUrl({ protocol: "http:", hostname: "[fe80::1]" }, 80),
    ).toBe("http://[fe80::1]:80/");
  });
});

describe("actionOutcome", () => {
  it("treats a controlled start failure as an error", () => {
    expect(
      actionOutcome({
        status: "unavailable",
        error: {
          code: "start_failed",
          message: "no GPU has room",
          type: "server_error",
        },
      }),
    ).toEqual({ kind: "error", message: "no GPU has room" });
    expect(actionOutcome({ status: "queue_full" })).toEqual({
      kind: "queue_full",
    });
  });

  it("treats everything else as success", () => {
    expect(actionOutcome({ status: "started", run_id: 1 })).toEqual({
      kind: "ok",
    });
    expect(actionOutcome({ status: "drained" })).toEqual({ kind: "ok" });
    expect(actionOutcome({ status: "already_disabled" })).toEqual({
      kind: "ok",
    });
  });
});
