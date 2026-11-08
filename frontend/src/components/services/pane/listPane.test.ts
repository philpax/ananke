import { describe, expect, it } from "vitest";

import {
  LIST_WIDTH_DEFAULT,
  LIST_WIDTH_MAX,
  LIST_WIDTH_MIN,
  clampListWidth,
  parseListPrefs,
  stepListWidth,
} from "./listPane.ts";

describe("clampListWidth", () => {
  it("keeps the width within bounds", () => {
    expect(clampListWidth(100)).toBe(LIST_WIDTH_MIN);
    expect(clampListWidth(10_000)).toBe(LIST_WIDTH_MAX);
    expect(clampListWidth(400.4)).toBe(400);
    expect(clampListWidth(Number.NaN)).toBe(LIST_WIDTH_DEFAULT);
  });
});

describe("stepListWidth", () => {
  it("steps with arrows, further with Shift", () => {
    expect(stepListWidth(400, "ArrowRight", false)).toBe(416);
    expect(stepListWidth(400, "ArrowLeft", true)).toBe(336);
  });

  it("jumps to the bounds with Home and End", () => {
    expect(stepListWidth(400, "Home", false)).toBe(LIST_WIDTH_MIN);
    expect(stepListWidth(400, "End", false)).toBe(LIST_WIDTH_MAX);
  });

  it("clamps at the edges and ignores other keys", () => {
    expect(stepListWidth(LIST_WIDTH_MAX, "ArrowRight", false)).toBe(
      LIST_WIDTH_MAX,
    );
    expect(stepListWidth(400, "Enter", false)).toBeNull();
  });
});

describe("parseListPrefs", () => {
  it("reads stored values", () => {
    expect(parseListPrefs("480", "true")).toEqual({
      width: 480,
      hidden: true,
    });
  });

  it("falls back on missing or malformed values", () => {
    expect(parseListPrefs(null, null)).toEqual({
      width: LIST_WIDTH_DEFAULT,
      hidden: false,
    });
    expect(parseListPrefs("wide", "yes")).toEqual({
      width: LIST_WIDTH_DEFAULT,
      hidden: false,
    });
    expect(parseListPrefs("9000", "false").width).toBe(LIST_WIDTH_MAX);
  });
});
