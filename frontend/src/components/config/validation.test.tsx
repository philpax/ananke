import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

import type { ValidationError } from "../../api/client.ts";
import { ValidationPanel } from "./ValidationPanel.tsx";
import { formatValidationError } from "./validation.ts";

function diagnostic(overrides: Partial<ValidationError> = {}): ValidationError {
  return {
    code: "value_invalid",
    message: "invalid value",
    path: "service.port",
    context: {
      kind: "Value",
      data: { field: "service.port", offending: "x", expected: "a port" },
    },
    location: null,
    line: null,
    column: null,
    ...overrides,
  };
}

describe("config validation diagnostics", () => {
  it("renders_located_parser_diagnostic", () => {
    const error = diagnostic({
      code: "parse",
      message: "parse error: expected a value",
      path: null,
      context: {
        kind: "Parse",
        data: { parser_message: "expected a value" },
      },
      location: { start: 4, end: 9, line: 2, column: 5 },
      line: 2,
      column: 5,
    });
    render(<ValidationPanel errors={[error]} />);
    expect(screen.getByText(/L2:5 \[parse\]/)).toBeInTheDocument();
    expect(screen.getByText(/expected a value/)).toBeInTheDocument();
  });

  it("renders_unlocated_semantic_diagnostic", () => {
    render(<ValidationPanel errors={[diagnostic()]} />);
    expect(
      screen.getByText(/\[value_invalid\] service\.port/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/L0:0/)).not.toBeInTheDocument();
  });

  it("renders_multiple_diagnostics_in_order", () => {
    render(
      <ValidationPanel
        errors={[
          diagnostic({ code: "field_missing", message: "first" }),
          diagnostic({ code: "value_invalid", message: "second" }),
        ]}
      />,
    );
    const text = screen.getByRole("list").textContent ?? "";
    expect(text.indexOf("first")).toBeLessThan(text.indexOf("second"));
  });

  it("renders_save_and_live_diagnostics_together", () => {
    render(
      <ValidationPanel
        errors={[
          diagnostic({ code: "parse", message: "live" }),
          diagnostic({ code: "merge_constraint", message: "save" }),
        ]}
      />,
    );
    expect(screen.getByText(/live/)).toBeInTheDocument();
    expect(screen.getByText(/save/)).toBeInTheDocument();
  });

  it("does_not_render_fake_zero_location", () => {
    render(<ValidationPanel errors={[diagnostic({ location: null })]} />);
    expect(screen.queryByText(/L0:0/)).not.toBeInTheDocument();
  });

  it("formatValidationError preserves pure response mapping", () => {
    expect(formatValidationError(diagnostic({ message: "mapped" }))).toContain(
      "mapped",
    );
  });
});
