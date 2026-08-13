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
    service: null,
    service_index: null,
    context: {
      kind: "Value",
      data: { field: "service.port", offending: "x", expected: "a port" },
    },
    location: null,
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

describe("formatContext covers every context kind", () => {
  it("names the service, falling back to its source index", () => {
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Service",
            data: { service: "alpha", index: 1, field: "port" },
          },
        }),
      ),
    ).toContain("service=alpha");
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Service",
            data: { service: null, index: 1, field: null },
          },
        }),
      ),
    ).toContain("service[1]");
  });

  it("pairs a field with its offending value when there is one", () => {
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Field",
            data: {
              field: "devices.split",
              offending: "diagonal",
              expected: null,
            },
          },
        }),
      ),
    ).toContain("devices.split=diagonal");
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Field",
            data: { field: "devices.split", offending: null, expected: null },
          },
        }),
      ),
    ).toContain("devices.split");
  });

  it("renders a count mismatch as got/expected", () => {
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Count",
            data: { field: "tensor_split_weights", got: 2, expected: 3 },
          },
        }),
      ),
    ).toContain("tensor_split_weights=2/3");
  });

  it("renders an indexed element with its position", () => {
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Index",
            data: {
              field: "tensor_split_weights",
              index: 1,
              value: "-1",
              expected: null,
            },
          },
        }),
      ),
    ).toContain("tensor_split_weights[1]=-1");
  });

  it("attributes a multi-field constraint to its service", () => {
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Fields",
            data: {
              fields: ["expert_offload", "devices.split"],
              service: "alpha",
              reason: "incompatible",
            },
          },
        }),
      ),
    ).toContain("service=alpha expert_offload, devices.split");
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Fields",
            data: {
              fields: ["expert_offload"],
              service: null,
              reason: "incompatible",
            },
          },
        }),
      ),
    ).toContain("expert_offload");
  });

  it("shows the offending placeholder argument when present", () => {
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Placeholder",
            data: {
              service: "alpha",
              field: "command",
              argv_index: 2,
              argument: "--port={nope}",
              category: "unknown",
            },
          },
        }),
      ),
    ).toContain("command=--port={nope}");
  });

  it("prefers the parent service over the child in merge context", () => {
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Merge",
            data: {
              service: "alpha",
              index: 0,
              parent: "base",
              reason: "cycle",
            },
          },
        }),
      ),
    ).toContain("parent=base");
    expect(
      formatValidationError(
        diagnostic({
          context: {
            kind: "Merge",
            data: { service: "alpha", index: 0, parent: null, reason: "cycle" },
          },
        }),
      ),
    ).toContain("service=alpha");
  });

  it("adds no chip for parser or forward-compatible payloads", () => {
    expect(
      formatValidationError(
        diagnostic({
          code: "parse",
          path: null,
          message: "boom",
          context: { kind: "Parse", data: { parser_message: "boom" } },
        }),
      ),
    ).toBe("[parse] boom");
    expect(
      formatValidationError(
        diagnostic({
          code: "other",
          path: null,
          message: "boom",
          context: { kind: "Other", data: { data: { future: true } } },
        }),
      ),
    ).toBe("[other] boom");
  });
});
