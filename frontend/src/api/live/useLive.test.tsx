import { act, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { DaemonInfoResponse } from "../client.ts";
import { FakeWebSocket } from "./fakeWebSocket.ts";
import type { LiveQueryDef } from "./resources.ts";
import { LiveSocket } from "./socket.ts";
import { LiveSocketContext, useLive } from "./useLive.ts";

function infoDef(fetch: () => Promise<DaemonInfoResponse>) {
  const def: LiveQueryDef<DaemonInfoResponse> = {
    queryKey: ["info"],
    resource: { kind: "info" },
    fetch,
    extract: (data) => (data.resource === "info" ? data.body : undefined),
  };
  return def;
}

function info(openai: string): DaemonInfoResponse {
  return {
    openai_listen: openai,
    management_listen: "127.0.0.1:7071",
    uptime_ms: 0,
  };
}

function Probe({ def }: { def: LiveQueryDef<DaemonInfoResponse> }) {
  const query = useLive(def);
  return <p>{query.data?.openai_listen ?? "pending"}</p>;
}

function renderProbe(def: LiveQueryDef<DaemonInfoResponse>) {
  const live = new LiveSocket({ url: () => "ws://test/api/live" });
  const qc = new QueryClient();
  render(
    <QueryClientProvider client={qc}>
      <LiveSocketContext.Provider value={live}>
        <Probe def={def} />
      </LiveSocketContext.Provider>
    </QueryClientProvider>,
  );
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

describe("useLive", () => {
  it("loads over HTTP, then takes snapshots from the socket", async () => {
    renderProbe(infoDef(() => Promise.resolve(info("from-http"))));
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(screen.getByText("from-http")).toBeInTheDocument();

    const ws = FakeWebSocket.latest();
    act(() => ws.accept());
    act(() =>
      ws.receive({
        type: "snapshot",
        id: "0",
        data: { resource: "info", body: info("from-socket") },
      }),
    );
    // TanStack batches cache notifications onto a timer.
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(screen.getByText("from-socket")).toBeInTheDocument();
  });

  it("polls only while the socket is down", async () => {
    const fetch = vi.fn(() => Promise.resolve(info("x")));
    renderProbe(infoDef(fetch));
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(fetch).toHaveBeenCalledTimes(1);

    await act(() => vi.advanceTimersByTimeAsync(5_000));
    expect(fetch).toHaveBeenCalledTimes(2);

    act(() => FakeWebSocket.latest().accept());
    await act(() => vi.advanceTimersByTimeAsync(20_000));
    expect(fetch).toHaveBeenCalledTimes(2);

    act(() => FakeWebSocket.latest().close());
    await act(() => vi.advanceTimersByTimeAsync(5_000));
    expect(fetch.mock.calls.length).toBeGreaterThan(2);
  });
});
