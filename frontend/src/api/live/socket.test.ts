import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { LiveData, LiveFeed } from "../client.ts";
import { FakeWebSocket } from "./fakeWebSocket.ts";
import {
  BACKOFF_INITIAL_MS,
  BACKOFF_MAX_MS,
  LINGER_MS,
  LiveSocket,
  backoffDelay,
} from "./socket.ts";

const INFO: LiveData = {
  resource: "info",
  body: {
    openai_listen: "127.0.0.1:7070",
    management_listen: "127.0.0.1:7071",
    uptime_ms: 0,
  },
};

function newSocket(): LiveSocket {
  // A jitter source of 0.5 makes every backoff delay exactly its base.
  return new LiveSocket({ url: () => "ws://test/api/live", random: () => 0.5 });
}

function listener() {
  return { onSnapshot: vi.fn(), onError: vi.fn() };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("WebSocket", FakeWebSocket);
  FakeWebSocket.reset();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("LiveSocket", () => {
  it("connects lazily and subscribes once the socket opens", () => {
    const live = newSocket();
    expect(FakeWebSocket.instances).toHaveLength(0);

    live.subscribe({ kind: "info" }, listener());
    const ws = FakeWebSocket.latest();
    expect(ws.sent).toEqual([]);
    expect(live.getStatus()).toBe("connecting");

    ws.accept();
    expect(live.getStatus()).toBe("open");
    expect(ws.sent).toEqual([
      { type: "subscribe", id: "0", resource: { kind: "info" } },
    ]);
  });

  it("delivers snapshots to the holder and ignores unknown ids", () => {
    const live = newSocket();
    const holder = listener();
    live.subscribe({ kind: "info" }, holder);
    const ws = FakeWebSocket.latest();
    ws.accept();

    ws.receive({ type: "snapshot", id: "99", data: INFO });
    expect(holder.onSnapshot).not.toHaveBeenCalled();

    ws.receive({ type: "snapshot", id: "0", data: INFO });
    expect(holder.onSnapshot).toHaveBeenCalledWith(INFO);

    // A later holder of the same resource gets the latest snapshot at once.
    const late = listener();
    live.subscribe({ kind: "info" }, late);
    expect(late.onSnapshot).toHaveBeenCalledWith(INFO);
  });

  it("routes resource errors to the holder", () => {
    const live = newSocket();
    const holder = listener();
    live.subscribe({ kind: "service", name: "ghost" }, holder);
    const ws = FakeWebSocket.latest();
    ws.accept();
    const error = {
      error: {
        code: "service_not_found" as const,
        message: "service `ghost` not found",
        type: "invalid_request_error" as const,
      },
    };
    ws.receive({ type: "error", id: "0", status: 404, error });
    expect(holder.onError).toHaveBeenCalledWith(error, 404);
  });

  it("resubscribes every held resource after reconnecting", () => {
    const live = newSocket();
    live.subscribe({ kind: "services" }, listener());
    live.subscribe({ kind: "devices" }, listener());
    const first = FakeWebSocket.latest();
    first.accept();

    first.close();
    expect(live.getStatus()).toBe("reconnecting");
    vi.advanceTimersByTime(BACKOFF_INITIAL_MS - 1);
    expect(FakeWebSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    const second = FakeWebSocket.latest();
    expect(second).not.toBe(first);

    second.accept();
    expect(second.sent).toEqual([
      { type: "subscribe", id: "0", resource: { kind: "services" } },
      { type: "subscribe", id: "1", resource: { kind: "devices" } },
    ]);
  });

  it("backs off exponentially up to the cap, and resets once open", () => {
    const live = newSocket();
    live.subscribe({ kind: "info" }, listener());
    const delays: number[] = [];
    for (let i = 0; i < 8; i++) {
      const before = FakeWebSocket.instances.length;
      FakeWebSocket.latest().close();
      let waited = 0;
      while (FakeWebSocket.instances.length === before) {
        vi.advanceTimersByTime(100);
        waited += 100;
      }
      delays.push(waited);
    }
    expect(delays).toEqual([500, 1000, 2000, 4000, 8000, 15000, 15000, 15000]);

    FakeWebSocket.latest().accept();
    FakeWebSocket.latest().close();
    vi.advanceTimersByTime(BACKOFF_INITIAL_MS);
    expect(FakeWebSocket.latest().closed).toBe(false);
  });

  it("jitters the delay by at most a fifth", () => {
    expect(backoffDelay(10, () => 0)).toBe(BACKOFF_MAX_MS * 0.8);
    expect(backoffDelay(10, () => 0.9999)).toBeLessThan(BACKOFF_MAX_MS * 1.2);
  });

  it("shares a resource between holders and cancels after the linger", () => {
    const live = newSocket();
    const releaseA = live.subscribe({ kind: "services" }, listener());
    const releaseB = live.subscribe({ kind: "services" }, listener());
    const ws = FakeWebSocket.latest();
    ws.accept();
    expect(ws.sent).toHaveLength(1);

    releaseA();
    releaseB();
    vi.advanceTimersByTime(LINGER_MS - 1);
    expect(ws.sent).toHaveLength(1);

    // Picking it back up within the linger keeps the server subscription.
    const releaseC = live.subscribe({ kind: "services" }, listener());
    vi.advanceTimersByTime(LINGER_MS);
    expect(ws.sent).toHaveLength(1);

    releaseC();
    vi.advanceTimersByTime(LINGER_MS);
    expect(ws.sent).toEqual([
      { type: "subscribe", id: "0", resource: { kind: "services" } },
      { type: "cancel", id: "0" },
    ]);
  });

  it("flags a daemon update only when the build changes across connections", () => {
    const live = newSocket();
    const changed = vi.fn();
    live.onChange(changed);
    live.subscribe({ kind: "info" }, listener());

    const first = FakeWebSocket.latest();
    first.accept();
    first.receive({ type: "hello", build: "0.2.0+aaaa" });
    first.close();
    vi.advanceTimersByTime(BACKOFF_INITIAL_MS);

    const second = FakeWebSocket.latest();
    second.accept();
    second.receive({ type: "hello", build: "0.2.0+aaaa" });
    expect(live.getDaemonUpdated()).toBe(false);
    second.close();
    vi.advanceTimersByTime(BACKOFF_INITIAL_MS);

    changed.mockClear();
    const third = FakeWebSocket.latest();
    third.receive({ type: "hello", build: "0.2.1+bbbb" });
    expect(live.getDaemonUpdated()).toBe(true);
    expect(changed).toHaveBeenCalledOnce();
  });

  it("keeps the first build it was told, for display", () => {
    const live = newSocket();
    const changed = vi.fn();
    live.onChange(changed);
    live.subscribe({ kind: "info" }, listener());
    expect(live.getBuild()).toBeNull();

    const socket = FakeWebSocket.latest();
    socket.accept();
    changed.mockClear();
    socket.receive({ type: "hello", build: "0.2.0+aaaa" });
    expect(live.getBuild()).toBe("0.2.0+aaaa");
    expect(changed).toHaveBeenCalledOnce();
  });

  it("reconnects on demand without waiting for the backoff", () => {
    const live = newSocket();
    live.subscribe({ kind: "info" }, listener());
    const first = FakeWebSocket.latest();
    first.accept();
    live.reconnect();
    expect(first.closed).toBe(true);
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(live.getStatus()).toBe("reconnecting");
  });

  describe("follows", () => {
    const FEED: LiveFeed = { kind: "logs", service: "demo", after: null };

    function follower(resume: () => LiveFeed = () => FEED) {
      return { resume: vi.fn(resume), onMessage: vi.fn() };
    }

    it("sends a follow once the socket opens", () => {
      const live = newSocket();
      live.follow(follower());
      const ws = FakeWebSocket.latest();
      expect(ws.sent).toEqual([]);
      ws.accept();
      expect(ws.sent).toEqual([{ type: "follow", id: "f0", feed: FEED }]);
    });

    it("resends with the holder's position after reconnecting", () => {
      const live = newSocket();
      let after: LiveFeed["after"] = null;
      live.follow(follower(() => ({ ...FEED, after })));
      const first = FakeWebSocket.latest();
      first.accept();
      after = { run_id: 3, seq: 9 };
      first.close();
      vi.advanceTimersByTime(BACKOFF_INITIAL_MS);

      const second = FakeWebSocket.latest();
      second.accept();
      expect(second.sent).toEqual([
        {
          type: "follow",
          id: "f0",
          feed: { ...FEED, after: { run_id: 3, seq: 9 } },
        },
      ]);
    });

    it("routes log frames and errors by id and drops unknown ids", () => {
      const live = newSocket();
      const a = follower();
      const b = follower();
      live.follow(a);
      live.follow(b);
      const ws = FakeWebSocket.latest();
      ws.accept();

      const frame = {
        type: "log_lines" as const,
        id: "f1",
        lines: [
          { timestamp_ms: 0, stream: "stdout", line: "hi", run_id: 1, seq: 1 },
        ],
      };
      ws.receive(frame);
      ws.receive({ type: "log_caught_up", id: "f0" });
      ws.receive({ type: "log_caught_up", id: "f9" });
      const error = {
        error: {
          code: "service_not_found" as const,
          message: "service `demo` not found",
          type: "invalid_request_error" as const,
        },
      };
      ws.receive({ type: "error", id: "f0", status: 404, error });

      expect(b.onMessage.mock.calls).toEqual([[frame]]);
      expect(a.onMessage.mock.calls).toEqual([
        [{ type: "log_caught_up", id: "f0" }],
        [{ type: "error", id: "f0", status: 404, error }],
      ]);
    });

    it("cancels at once on release and ignores later frames", () => {
      const live = newSocket();
      const holder = follower();
      const release = live.follow(holder);
      const ws = FakeWebSocket.latest();
      ws.accept();
      release();
      expect(ws.sent.at(-1)).toEqual({ type: "cancel", id: "f0" });
      ws.receive({ type: "log_caught_up", id: "f0" });
      expect(holder.onMessage).not.toHaveBeenCalled();
    });
  });
});
