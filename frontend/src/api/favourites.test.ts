import { describe, expect, it } from "vitest";

import { parseFavourites } from "./favourites.ts";

describe("parseFavourites", () => {
  it("keeps the strings of a stored array", () => {
    expect(parseFavourites(["a", 3, "b", null])).toEqual(["a", "b"]);
  });

  it("treats anything else as no favourites", () => {
    expect(parseFavourites({ a: true })).toEqual([]);
    expect(parseFavourites("a")).toEqual([]);
    expect(parseFavourites(null)).toEqual([]);
  });
});
