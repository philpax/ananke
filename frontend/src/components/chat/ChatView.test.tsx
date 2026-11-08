import { fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  installLiveFakes,
  renderLive,
  service,
} from "../../test/renderLive.tsx";
import { ChatView } from "./ChatView.tsx";
import { getSnapshot, setSnapshot, type Message } from "./state/chatStore.ts";
import { resetChat } from "./state/chatTestKit.ts";

beforeEach(() => {
  installLiveFakes();
  localStorage.clear();
  resetChat();
});
afterEach(() => vi.unstubAllGlobals());

const services = [service("alpha", "running"), service("beta", "running")];

describe("ChatView", () => {
  it("switches the model mid-conversation", () => {
    setSnapshot((prev) => ({
      ...prev,
      currentModel: "alpha",
      messages: [{ id: 1, role: "user", content: "hello", timestamp: "t1" }],
    }));
    renderLive(<ChatView />, { route: "/chat?model=alpha", services });

    fireEvent.click(screen.getByRole("button", { name: "alpha" }));
    fireEvent.click(screen.getByRole("button", { name: "beta" }));

    expect(getSnapshot().currentModel).toBe("beta");
    expect(getSnapshot().messages).toHaveLength(1);
  });

  it("follows a model named in the URL", () => {
    renderLive(<ChatView />, { route: "/chat?model=beta", services });
    expect(getSnapshot().currentModel).toBe("beta");
  });

  it("shows a model's start as a wait that New chat ends", () => {
    const turn: Message = {
      id: 1,
      role: "user",
      content: "hello",
      timestamp: "t",
    };
    const controller = new AbortController();
    setSnapshot((prev) => ({
      ...prev,
      currentModel: "alpha",
      messages: [turn],
      chatState: {
        kind: "starting",
        run: {
          controller,
          model: "alpha",
          turn,
          draft: { input: "hello", attachments: [] },
        },
      },
    }));
    renderLive(<ChatView />, { route: "/chat?model=alpha", services });
    expect(screen.getByText("Starting alpha…")).toBeVisible();
    expect(screen.getByText(/1 exchange$/)).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    expect(controller.signal.aborted).toBe(true);
    expect(screen.queryByText("Starting alpha…")).not.toBeInTheDocument();
    expect(screen.queryByText("hello")).not.toBeInTheDocument();
  });
});
