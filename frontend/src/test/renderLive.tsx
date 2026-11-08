// Renders a component the way the app does: under a router, a query
// client, and a live socket. The socket is real but talks to
// `FakeWebSocket`, and HTTP never answers, so the data a test seeds is the
// data the component sees. Install the fakes with `installLiveFakes` in a
// `beforeEach`.

import type { ReactNode } from "react";
import { act, render, type RenderResult } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import type {
  DaemonInfoResponse,
  DeviceSummary,
  ServiceStateKind,
  ServiceSummary,
} from "../api/client.ts";
import { FakeWebSocket } from "../api/live/fakeWebSocket.ts";
import { LiveSocket } from "../api/live/socket.ts";
import { LiveSocketContext } from "../api/live/useLive.ts";

export type LiveSeed = {
  readonly route?: string;
  readonly services?: readonly ServiceSummary[];
  readonly devices?: readonly DeviceSummary[];
  readonly info?: DaemonInfoResponse;
};

export type LiveRender = RenderResult & {
  readonly live: LiveSocket;
  readonly client: QueryClient;
};

export function installLiveFakes(): void {
  FakeWebSocket.reset();
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal(
    "fetch",
    vi.fn(() => new Promise<never>(() => {})),
  );
}

export function renderLive(ui: ReactNode, seed: LiveSeed = {}): LiveRender {
  const live = new LiveSocket({ url: () => "ws://test/api/live" });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  if (seed.services !== undefined) {
    client.setQueryData(["services"], {
      openai_api_port: 7070,
      services: seed.services,
    });
  }
  if (seed.devices !== undefined)
    client.setQueryData(["devices"], seed.devices);
  if (seed.info !== undefined) client.setQueryData(["info"], seed.info);
  const result = render(
    <QueryClientProvider client={client}>
      <LiveSocketContext.Provider value={live}>
        <MemoryRouter initialEntries={[seed.route ?? "/"]}>{ui}</MemoryRouter>
      </LiveSocketContext.Provider>
    </QueryClientProvider>,
  );
  return { ...result, live, client };
}

/// Replaces the services list, as a live snapshot would. TanStack defers
/// its notifications to a timer, so this waits one out.
export async function setServices(
  client: QueryClient,
  services: readonly ServiceSummary[],
): Promise<void> {
  await act(async () => {
    client.setQueryData(["services"], { openai_api_port: 7070, services });
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

export function service(
  name: string,
  state: ServiceStateKind = "idle",
  extra: Partial<ServiceSummary> = {},
): ServiceSummary {
  return {
    name,
    state,
    lifecycle: "on_demand",
    openai_compat: true,
    port: 8000,
    priority: 50,
    ...extra,
  };
}
