import { act, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FakeWebSocket } from "../../api/live/fakeWebSocket.ts";
import { installLiveFakes, renderLive } from "../../test/renderLive.tsx";
import { ConnectionStatus } from "./ConnectionStatus.tsx";

beforeEach(() => {
  vi.useFakeTimers();
  installLiveFakes();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("ConnectionStatus", () => {
  it("follows the socket from connecting to open to reconnecting", () => {
    const { live } = renderLive(<ConnectionStatus />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Connecting");

    act(() => {
      live.subscribe({ kind: "info" }, { onSnapshot() {}, onError() {} });
    });
    const socket = FakeWebSocket.latest();
    act(() => socket.accept());
    expect(status).toHaveTextContent("Connected");

    act(() => socket.close());
    expect(status).toHaveTextContent(
      "Reconnecting. Showing the last known state.",
    );
  });
});
