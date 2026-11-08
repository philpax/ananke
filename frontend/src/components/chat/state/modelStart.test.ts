import { describe, expect, it } from "vitest";

import { nextStep, type StartRequest } from "./modelStart.ts";

const pending: StartRequest = { kind: "pending" };
const cut: StartRequest = { kind: "rejected", message: "504" };

describe("nextStep", () => {
  it("is ready once the service runs, whatever the request says", () => {
    expect(nextStep(pending, "running")).toEqual({ kind: "ready" });
    expect(nextStep(cut, "running")).toEqual({ kind: "ready" });
  });

  it("waits on a pending request, even through a failed poll", () => {
    expect(nextStep(pending, "starting")).toEqual({
      kind: "wait",
      reissue: false,
    });
    expect(nextStep(pending, undefined)).toEqual({
      kind: "wait",
      reissue: false,
    });
  });

  it("reissues a lost request while the service is starting", () => {
    expect(nextStep(cut, "starting")).toEqual({ kind: "wait", reissue: true });
  });

  it("believes a rejection once the service is not starting", () => {
    for (const state of ["failed", "idle", "disabled", undefined] as const) {
      expect(nextStep(cut, state)).toEqual({
        kind: "failed",
        failure: { kind: "error", message: "504" },
      });
    }
  });

  it("reads the start's answer", () => {
    const answer = (response: StartRequest & { kind: "answered" }) =>
      nextStep(response, "starting");
    expect(
      answer({ kind: "answered", response: { status: "started", run_id: 1 } }),
    ).toEqual({ kind: "ready" });
    expect(
      answer({ kind: "answered", response: { status: "queue_full" } }),
    ).toEqual({ kind: "failed", failure: { kind: "queue_full" } });
  });
});
