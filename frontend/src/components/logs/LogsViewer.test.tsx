// Container output arrives on a third stream. These check that the viewer
// treats it as its own thing: `both` includes it, `stdout` does not sweep it
// up, and it renders under its own marker rather than being mistaken for
// the child's stdout.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { LogLine } from "../../api/client.ts";
import { LogRow } from "./LogRow.tsx";
import { filterLines } from "./logFilter.ts";

function line(stream: string, text: string, run_id = 1, seq = 1): LogLine {
  return { timestamp_ms: 1_700_000_000_000, stream, line: text, run_id, seq };
}

const buffer: readonly LogLine[] = [
  line("stdout", "child stdout", 1, 1),
  line("stderr", "child stderr", 1, 2),
  line("combined", "container output", 2, 3),
];

describe("logs_viewer_filters_and_labels_combined", () => {
  it("includes every stream under `both`", () => {
    expect(filterLines(buffer, "both", null, "")).toHaveLength(3);
  });

  it("selects only container output under `combined`", () => {
    const out = filterLines(buffer, "combined", null, "");
    expect(out).toHaveLength(1);
    expect(out[0]!.line).toBe("container output");
  });

  it("does not fold container output into stdout or stderr", () => {
    expect(filterLines(buffer, "stdout", null, "")).toEqual([buffer[0]]);
    expect(filterLines(buffer, "stderr", null, "")).toEqual([buffer[1]]);
  });

  it("composes with the run and search filters", () => {
    expect(filterLines(buffer, "combined", 2, "")).toHaveLength(1);
    expect(filterLines(buffer, "combined", 1, "")).toHaveLength(0);
    expect(filterLines(buffer, "both", null, "CONTAINER")).toHaveLength(1);
  });

  it("labels each stream distinctly", () => {
    render(<LogRow line={line("combined", "container output")} search="" />);
    expect(screen.getByText("combined")).toBeInTheDocument();
    expect(screen.queryByText("out")).not.toBeInTheDocument();
    expect(screen.queryByText("err")).not.toBeInTheDocument();
  });

  it("keeps the neutral container marker away from the error styling", () => {
    // stderr is the only stream that reads as a problem; a container's
    // merged output carries ordinary informational lines too.
    const { container: combined } = render(
      <LogRow line={line("combined", "loading")} search="" />,
    );
    expect(combined.querySelector(".text-danger")).toBeNull();

    const { container: err } = render(
      <LogRow line={line("stderr", "boom")} search="" />,
    );
    expect(err.querySelector(".text-danger")).not.toBeNull();
  });
});
