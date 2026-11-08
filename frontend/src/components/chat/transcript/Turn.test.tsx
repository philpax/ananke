import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { EMPTY_STATS, type Message } from "../state/chatStore.ts";
import { Turn } from "./Turn.tsx";

function reply(fields: Partial<Message> = {}): Message {
  return {
    id: 1,
    role: "assistant",
    content: "",
    model: "qwen",
    timestamp: "2026-10-23T14:30:05+02:00",
    ...fields,
  };
}

/// jsdom has no layout, so the trace's scroll box is given one.
function layOut(el: HTMLElement, box: { height: number }): void {
  Object.defineProperty(el, "clientHeight", { get: () => 100 });
  Object.defineProperty(el, "scrollHeight", { get: () => box.height });
}

describe("Turn", () => {
  it("labels a reply with its model, time, and token figures", () => {
    render(
      <Turn
        message={reply({ content: "hi" })}
        liveStats={{
          ...EMPTY_STATS,
          promptTokens: 12,
          completionTokens: 3,
          predictedPerSecond: 30,
        }}
      />,
    );
    expect(screen.getByRole("heading", { name: "qwen" })).toBeVisible();
    expect(screen.getByText("14:30:05")).toBeVisible();
    expect(screen.getByText("12 in · 3 out · 30.0 tok/s")).toBeVisible();
  });

  it("copies a code block's source with its language shown", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    render(
      <Turn
        message={reply({ content: "```rust\nfn main() {}\n```" })}
        liveStats={null}
      />,
    );
    expect(screen.getByText("rust")).toBeVisible();
    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", { name: "Copy the code to clipboard" }),
      );
    });
    expect(writeText).toHaveBeenCalledWith("fn main() {}");
  });

  it("follows the reasoning trace until the reader scrolls up in it", () => {
    const { rerender } = render(
      <Turn message={reply({ reasoning: "a" })} liveStats={null} />,
    );
    const trace = screen.getByRole("region", { name: "Reasoning" });
    const box = { height: 500 };
    layOut(trace, box);

    const grow = (reasoning: string, height: number) => {
      box.height = height;
      rerender(<Turn message={reply({ reasoning })} liveStats={null} />);
    };

    grow("a b", 500);
    expect(trace.scrollTop).toBe(500);

    trace.scrollTop = 100;
    fireEvent.scroll(trace);
    grow("a b c", 800);
    expect(trace.scrollTop).toBe(100);

    trace.scrollTop = 700;
    fireEvent.scroll(trace);
    grow("a b c d", 900);
    expect(trace.scrollTop).toBe(900);
  });
});
