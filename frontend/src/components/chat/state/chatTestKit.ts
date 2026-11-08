// Fakes for driving `send` without a daemon: a scripted `fetch` that
// answers the services list, the start request, and the chat stream, and
// a store reset between tests.

import { vi } from "vitest";

import type { ServiceStateKind } from "../../../api/client.ts";
import { service } from "../../../test/renderLive.tsx";
import { EMPTY_STATS, setSnapshot } from "./chatStore.ts";

export const MODEL = "m";
export const BASE_URL = "http://openai.test";

export function resetChat(): void {
  setSnapshot(() => ({
    messages: [],
    systemPrompt: "",
    currentModel: null,
    chatState: { kind: "idle" },
    stats: EMPTY_STATS,
    input: "",
    attachments: [],
  }));
}

export type Call = { readonly url: string; readonly method: string };

/// Answers one request. Return `undefined` to use the default answer.
export type Handler = (
  call: Call,
  index: number,
) => Promise<Response> | Response | undefined;

export type FakeDaemon = {
  readonly calls: Call[];
  /// The state each services poll reports, in order; the last repeats.
  states: ServiceStateKind[];
};

export function installFakeDaemon(
  states: ServiceStateKind[],
  handler: Handler = () => undefined,
): FakeDaemon {
  const daemon: FakeDaemon = { calls: [], states };
  let polls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const call = {
        url: input instanceof Request ? input.url : String(input),
        method: init?.method ?? "GET",
      };
      daemon.calls.push(call);
      const custom = handler(call, daemon.calls.length);
      if (custom !== undefined) return Promise.resolve(custom);
      if (call.url === "/api/services") {
        const state =
          daemon.states[Math.min(polls, daemon.states.length - 1)] ?? "idle";
        polls += 1;
        return Promise.resolve(
          Response.json({
            openai_api_port: 7070,
            services: [service(MODEL, state)],
          }),
        );
      }
      if (call.url.endsWith("/start")) {
        return Promise.resolve(
          Response.json({ status: "started", run_id: 1 }, { status: 202 }),
        );
      }
      if (call.url === `${BASE_URL}/v1/chat/completions`) {
        return Promise.resolve(sse(["reply"]));
      }
      return Promise.reject(new TypeError(`unexpected fetch ${call.url}`));
    }),
  );
  return daemon;
}

/// A chat stream sending each piece of content as a delta, then `[DONE]`.
export function sse(
  content: readonly string[],
  { done = true }: { readonly done?: boolean } = {},
): Response {
  const frames = content.map(
    (c) => `data: ${JSON.stringify({ choices: [{ delta: { content: c } }] })}`,
  );
  if (done) frames.push("data: [DONE]");
  return new Response(frames.map((f) => `${f}\n\n`).join(""), {
    headers: { "content-type": "text/event-stream" },
  });
}

/// Runs `promise` to completion, advancing fake timers as it waits.
export async function settle<T>(promise: Promise<T>): Promise<T> {
  let done = false;
  const result = promise.finally(() => {
    done = true;
  });
  for (let i = 0; i < 1000 && !done; i++) {
    await vi.advanceTimersByTimeAsync(500);
  }
  return result;
}
