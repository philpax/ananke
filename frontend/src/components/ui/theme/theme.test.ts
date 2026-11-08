import { describe, expect, it } from "vitest";

import { nextOverride, parseOverride, resolveTheme } from "./theme.ts";

describe("parseOverride", () => {
  it("accepts only light and dark", () => {
    expect(parseOverride("light")).toBe("light");
    expect(parseOverride("dark")).toBe("dark");
    expect(parseOverride("system")).toBeNull();
    expect(parseOverride(null)).toBeNull();
  });
});

describe("resolveTheme", () => {
  it("follows the OS without an override", () => {
    expect(resolveTheme(null, true)).toBe("dark");
    expect(resolveTheme(null, false)).toBe("light");
  });

  it("prefers the override to the OS", () => {
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
  });
});

describe("nextOverride", () => {
  it("stores the opposite of the OS when following it", () => {
    expect(nextOverride(null, true)).toBe("light");
    expect(nextOverride(null, false)).toBe("dark");
  });

  it("clears the override when toggling back to the OS theme", () => {
    expect(nextOverride("light", true)).toBeNull();
    expect(nextOverride("dark", false)).toBeNull();
  });

  it("flips an override the OS has since come to match", () => {
    // The OS changed to dark after the user pinned dark; the page shows
    // dark, so one press goes to light, which the OS does not ask for.
    expect(nextOverride("dark", true)).toBe("light");
  });
});
