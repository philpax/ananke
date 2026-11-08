import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { LogToolbar, type LogStatus } from "./LogToolbar.tsx";

function renderToolbar(status: Partial<LogStatus>) {
  render(
    <LogToolbar
      status={{
        fixed: false,
        live: false,
        skipped: 0,
        truncated: false,
        lineCount: 3,
        ...status,
      }}
      stream="both"
      onStream={() => {}}
      search=""
      onSearch={() => {}}
      runIds={[]}
      run={null}
      onRun={() => {}}
      pinned
      onFollow={() => {}}
    />,
  );
}

describe("LogToolbar", () => {
  it("labels a fixed range as selected and does not offer to follow", () => {
    renderToolbar({ fixed: true, live: true });
    expect(screen.getByText("selected range")).toBeInTheDocument();
    expect(screen.queryByText("live")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to end" })).toBeVisible();
  });

  it("shows a rolling range as live once caught up", () => {
    renderToolbar({ live: true });
    expect(screen.getByText("live")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "follow" })).toBeVisible();
  });
});
