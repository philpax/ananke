import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Chip } from "./Chip.tsx";
import { StateChip } from "./StateChip.tsx";

describe("Chip", () => {
  it("renders its children", () => {
    render(<Chip>vision</Chip>);
    expect(screen.getByText("vision")).toBeInTheDocument();
  });

  it("takes the status tone's colours", () => {
    render(<Chip tone="ok">ok</Chip>);
    expect(screen.getByText("ok")).toHaveClass("text-ok", "bg-ok-bg");
  });

  it("lets a caller's class override the default", () => {
    render(
      <Chip tone="vision" className="px-2">
        vision
      </Chip>,
    );
    const chip = screen.getByText("vision");
    expect(chip).toHaveClass("px-2");
    expect(chip).not.toHaveClass("px-1");
  });
});

describe("StateChip", () => {
  it("translates the state and colours it by tone", () => {
    render(<StateChip state="running" />);
    expect(screen.getByText("running")).toHaveClass("text-ok");
  });

  it("labels any disabled state as disabled", () => {
    render(<StateChip state="disabled" />);
    expect(screen.getByText("disabled")).toHaveClass("text-body");
  });
});
