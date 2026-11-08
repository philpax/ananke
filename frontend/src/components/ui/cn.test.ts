import { describe, expect, it } from "vitest";

import { cn } from "./cn.ts";

describe("cn", () => {
  it("keeps a custom font size beside a text colour", () => {
    expect(cn("text-2xs", "text-muted")).toBe("text-2xs text-muted");
  });

  it("lets a later named control height override an earlier one", () => {
    expect(cn("h-control", "h-control-sm")).toBe("h-control-sm");
    expect(cn("h-8", "h-control")).toBe("h-control");
  });

  it("lets a later colour override an earlier one", () => {
    expect(cn("text-soft", "text-ink")).toBe("text-ink");
  });
});
