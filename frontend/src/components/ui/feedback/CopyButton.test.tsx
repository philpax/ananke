import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CopyButton } from "./CopyButton.tsx";

function stubClipboard(writeText: (text: string) => Promise<void>) {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
}

afterEach(() => {
  vi.useRealTimers();
});

describe("CopyButton", () => {
  it("names what it copies", () => {
    render(<CopyButton value="abc123" label="the run id" />);
    expect(
      screen.getByRole("button", { name: "Copy the run id to clipboard" }),
    ).toBeInTheDocument();
  });

  it("copies the value and announces success", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    stubClipboard(writeText);
    render(<CopyButton value="abc123" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button"));
    });
    expect(writeText).toHaveBeenCalledWith("abc123");
    expect(screen.getByRole("status")).toHaveTextContent("Copied");
  });

  it("announces a clipboard that is unavailable", async () => {
    stubClipboard(() => Promise.reject(new Error("denied")));
    render(<CopyButton value="abc123" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button"));
    });
    expect(screen.getByRole("status")).toHaveTextContent(
      "The clipboard is unavailable.",
    );
  });

  it("does not let the click reach a surrounding row", async () => {
    stubClipboard(() => Promise.resolve());
    const onRowClick = vi.fn();
    render(
      <div onClick={onRowClick}>
        <CopyButton value="abc123" />
      </div>,
    );
    await act(async () => {
      fireEvent.click(screen.getByRole("button"));
    });
    expect(onRowClick).not.toHaveBeenCalled();
  });

  it("goes back to idle after a while", async () => {
    vi.useFakeTimers();
    stubClipboard(() => Promise.resolve());
    render(<CopyButton value="abc123" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button"));
    });
    expect(screen.getByRole("status")).toHaveTextContent("Copied");
    act(() => {
      vi.advanceTimersByTime(3_000);
    });
    expect(screen.getByRole("status")).toHaveTextContent("");
  });
});
