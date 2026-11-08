import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  cancel,
  clearConversation,
  getSnapshot,
  send,
  setInput,
} from "./chatStore.ts";
import {
  BASE_URL,
  MODEL,
  installFakeDaemon,
  resetChat,
  settle,
  sse,
} from "./chatTestKit.ts";

beforeEach(() => {
  vi.useFakeTimers();
  resetChat();
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("send", () => {
  it("adds the message and the reply", async () => {
    installFakeDaemon(["running"]);
    setInput("hello");
    await settle(send(MODEL, BASE_URL));
    const { messages, chatState, input } = getSnapshot();
    expect(messages.map((m) => [m.role, m.content])).toEqual([
      ["user", "hello"],
      ["assistant", "reply"],
    ]);
    expect(chatState.kind).toBe("idle");
    expect(input).toBe("");
  });

  // A reset at any point of a send, starting a stopped model on the way,
  // must leave the conversation empty: nothing the send was still waiting
  // on may write the message back.
  it.each([1, 2, 3])(
    "is undone by a reset during request %i",
    async (resetAt) => {
      const daemon = installFakeDaemon(["stopped", "running"], (_, index) => {
        if (index === resetAt) clearConversation();
        return undefined;
      });
      setInput("hello");
      await settle(send(MODEL, BASE_URL));
      expect(daemon.calls.length).toBeGreaterThanOrEqual(resetAt);
      expect(getSnapshot().messages).toEqual([]);
      expect(getSnapshot().chatState.kind).toBe("idle");
    },
  );

  it("ignores a second send while the first is in flight", async () => {
    const daemon = installFakeDaemon(["running"]);
    setInput("hello");
    const first = send(MODEL, BASE_URL);
    setInput("again");
    const second = send(MODEL, BASE_URL);
    await settle(Promise.all([first, second]));
    const sent = daemon.calls.filter((c) => c.url.startsWith(BASE_URL));
    expect(sent).toHaveLength(1);
    expect(getSnapshot().messages.map((m) => m.content)).toEqual([
      "hello",
      "reply",
    ]);
  });

  // The start request is held open until the model is ready, so a slow
  // load can outlast a proxy or the browser and come back as a 504 or a
  // network error while the daemon carries on starting it.
  it("keeps waiting when a slow start's request is cut", async () => {
    let starts = 0;
    installFakeDaemon(
      ["stopped", "starting", "starting", "starting", "running"],
      (call) => {
        if (!call.url.endsWith("/start")) return undefined;
        starts += 1;
        return starts === 1
          ? new Response("<html>Gateway Timeout</html>", { status: 504 })
          : new Promise<never>(() => {});
      },
    );
    setInput("hello");
    await settle(send(MODEL, BASE_URL));
    const { messages, chatState } = getSnapshot();
    expect(chatState.kind).toBe("idle");
    expect(messages.map((m) => m.content)).toEqual(["hello", "reply"]);
  });

  it("waits for as long as the model takes", async () => {
    const starting = Array.from({ length: 400 }, () => "starting" as const);
    installFakeDaemon(["stopped", ...starting, "running"], (call) =>
      call.url.endsWith("/start") ? new Promise<never>(() => {}) : undefined,
    );
    setInput("hello");
    const sending = send(MODEL, BASE_URL);
    await vi.advanceTimersByTimeAsync(5 * 60 * 1000);
    const { messages, chatState } = getSnapshot();
    expect(chatState.kind).toBe("starting");
    expect(messages.map((m) => m.content)).toEqual(["hello"]);
    await settle(sending);
    expect(getSnapshot().messages.map((m) => m.content)).toEqual([
      "hello",
      "reply",
    ]);
  });

  it("reports a start the daemon declines and gives the text back", async () => {
    installFakeDaemon(["stopped"], (call) =>
      call.url.endsWith("/start")
        ? Response.json(
            {
              status: "unavailable",
              error: {
                code: "insufficient_capacity",
                message: "not enough VRAM",
                type: "server_error",
              },
            },
            { status: 202 },
          )
        : undefined,
    );
    setInput("hello");
    await settle(send(MODEL, BASE_URL));
    const { messages, chatState, input } = getSnapshot();
    expect(chatState).toEqual({
      kind: "error",
      message: "Failed to start m: not enough VRAM",
    });
    expect(messages).toEqual([]);
    expect(input).toBe("hello");
  });

  it("reports a start that fails once the model stops starting", async () => {
    installFakeDaemon(["stopped", "failed"], (call) =>
      call.url.endsWith("/start")
        ? Response.json(
            {
              error: {
                code: "start_failed",
                message: "health check timed out",
                type: "server_error",
              },
            },
            { status: 503 },
          )
        : undefined,
    );
    setInput("hello");
    await settle(send(MODEL, BASE_URL));
    expect(getSnapshot().chatState).toEqual({
      kind: "error",
      message: "Failed to start m: health check timed out",
    });
  });

  it("fails a reply whose stream stops before [DONE]", async () => {
    installFakeDaemon(["running"], (call) =>
      call.url.startsWith(BASE_URL)
        ? sse(["half a"], { done: false })
        : undefined,
    );
    setInput("hello");
    await settle(send(MODEL, BASE_URL));
    const { messages, chatState } = getSnapshot();
    expect(chatState).toEqual({
      kind: "error",
      message: "The reply ended before the model finished it.",
    });
    expect(messages.map((m) => m.content)).toEqual(["hello", "half a"]);
  });

  it("gives the text back when cancelled before a reply", async () => {
    installFakeDaemon(["running"], (call) =>
      call.url.startsWith(BASE_URL) ? new Promise<never>(() => {}) : undefined,
    );
    setInput("hello");
    const sending = send(MODEL, BASE_URL);
    await vi.advanceTimersByTimeAsync(0);
    expect(getSnapshot().chatState.kind).toBe("sending");
    cancel();
    expect(getSnapshot().messages).toEqual([]);
    expect(getSnapshot().input).toBe("hello");
    expect(getSnapshot().chatState.kind).toBe("idle");
    void sending;
  });
});
