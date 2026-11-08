import { describe, expect, it } from "vitest";

import { rowVariants } from "./rowVariants.ts";

describe("rowVariants", () => {
  it("draws the severity rail on an unselected row", () => {
    expect(rowVariants({ severity: "danger" }).split(" ")).toContain(
      "border-l-danger",
    );
  });

  it("replaces the severity rail with the accent when selected", () => {
    const classes = rowVariants({ severity: "danger", selected: true }).split(
      " ",
    );
    expect(classes).toContain("border-l-accent");
    expect(classes).not.toContain("border-l-danger");
  });
});
