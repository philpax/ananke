// Waits for a stopped model to come up before a chat request goes to it.
//
// `POST /start` is held open until the model is ready, which for a large
// model can take minutes; a proxy or the browser may cut it long before
// the daemon gives up. So the request's answer is believed when it is
// one, and the service's state is polled alongside it: a request cut
// while the service is still starting is simply issued again, and only
// the user (by cancelling) or the daemon (by a failure) ends the wait.

import type { ServiceStateKind, StartResponse } from "../../../api/client.ts";
import { api } from "../../../api/client.ts";

export type StartRequest =
  | { readonly kind: "pending" }
  | { readonly kind: "answered"; readonly response: StartResponse }
  | { readonly kind: "rejected"; readonly message: string };

export type StartFailure =
  | { readonly kind: "queue_full" }
  | { readonly kind: "error"; readonly message: string };

export type WaitStep =
  | { readonly kind: "ready" }
  | { readonly kind: "failed"; readonly failure: StartFailure }
  | { readonly kind: "wait"; readonly reissue: boolean };

/// Decides the wait from the start request so far and the service's
/// polled state (`undefined` when the poll failed).
export function nextStep(
  request: StartRequest,
  state: ServiceStateKind | undefined,
): WaitStep {
  if (state === "running") return { kind: "ready" };
  switch (request.kind) {
    case "pending":
      return { kind: "wait", reissue: false };
    case "answered":
      return answered(request.response);
    case "rejected":
      // A failed start leaves the service failed, idle, or disabled; one
      // still starting means only the request was lost.
      return state === "starting"
        ? { kind: "wait", reissue: true }
        : {
            kind: "failed",
            failure: { kind: "error", message: request.message },
          };
    default: {
      const unreachable: never = request;
      return unreachable;
    }
  }
}

export type WaitOutcome =
  | { readonly kind: "ready" }
  | { readonly kind: "failed"; readonly failure: StartFailure }
  | { readonly kind: "aborted" };

export type StartDeps = {
  readonly start: (name: string) => Promise<StartResponse>;
  readonly state: (name: string) => Promise<ServiceStateKind | undefined>;
  readonly sleep: (ms: number) => Promise<void>;
};

export const POLL_MS = 1000;

export async function waitForModel(
  name: string,
  signal: AbortSignal,
  deps: StartDeps = liveDeps,
): Promise<WaitOutcome> {
  let request: StartRequest = { kind: "pending" };
  let settled: Promise<unknown> = Promise.resolve();
  let generation = 0;
  const issue = () => {
    const mine = ++generation;
    request = { kind: "pending" };
    settled = deps.start(name).then(
      (response) => {
        if (mine === generation) request = { kind: "answered", response };
      },
      (e: unknown) => {
        if (mine === generation) {
          request = {
            kind: "rejected",
            message: e instanceof Error ? e.message : String(e),
          };
        }
      },
    );
  };

  issue();
  for (;;) {
    // A start that answers wakes the loop rather than waiting out a poll.
    await Promise.race([deps.sleep(POLL_MS), settled]);
    if (signal.aborted) return { kind: "aborted" };
    const state = await deps.state(name).catch(() => undefined);
    if (signal.aborted) return { kind: "aborted" };
    const step = nextStep(request, state);
    if (step.kind !== "wait") return step;
    if (step.reissue) issue();
  }
}

function answered(response: StartResponse): WaitStep {
  switch (response.status) {
    case "already_running":
    case "started":
      return { kind: "ready" };
    case "queue_full":
      return { kind: "failed", failure: { kind: "queue_full" } };
    case "unavailable":
      return {
        kind: "failed",
        failure: { kind: "error", message: response.error.message },
      };
    default: {
      const unreachable: never = response;
      return unreachable;
    }
  }
}

const liveDeps: StartDeps = {
  start: (name) => api.start(name),
  state: async (name) =>
    (await api.listServices()).services.find((s) => s.name === name)?.state,
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
};
