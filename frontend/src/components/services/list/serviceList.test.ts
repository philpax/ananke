import { describe, expect, it } from "vitest";

import { service } from "../../../test/renderLive.tsx";
import {
  filterServices,
  groupByState,
  parseSortOrder,
  sortServices,
} from "./serviceList.ts";

const none = new Set<string>();

describe("filterServices", () => {
  const services = [
    service("qwen-coder", "running", { description: "Code model" }),
    service("bge", "idle", { description: "Embeddings" }),
    service("llama", "idle"),
  ];

  it("matches name or description, ignoring case", () => {
    const names = (query: string) =>
      filterServices(services, {
        query,
        favouritesOnly: false,
        favourites: none,
      }).map((s) => s.name);
    expect(names("QWEN")).toEqual(["qwen-coder"]);
    expect(names("embed")).toEqual(["bge"]);
    expect(names("  ")).toHaveLength(3);
  });

  it("keeps only favourites when asked", () => {
    expect(
      filterServices(services, {
        query: "",
        favouritesOnly: true,
        favourites: new Set(["llama"]),
      }).map((s) => s.name),
    ).toEqual(["llama"]);
  });
});

describe("sortServices", () => {
  const services = [
    service("c", "idle", { last_used_ms: 30, footprint_bytes: 1 }),
    service("a", "idle", { last_used_ms: 10 }),
    service("b", "idle", { last_used_ms: 20, footprint_bytes: 9 }),
  ];
  const names = (order: "alpha" | "recent" | "size", favs = none) =>
    sortServices(services, order, favs).map((s) => s.name);

  it("sorts by name, recency, or size", () => {
    expect(names("alpha")).toEqual(["a", "b", "c"]);
    expect(names("recent")).toEqual(["c", "b", "a"]);
    expect(names("size")).toEqual(["b", "c", "a"]);
  });

  it("puts favourites first", () => {
    expect(names("alpha", new Set(["c"]))).toEqual(["c", "a", "b"]);
  });

  it("falls back to alpha for an unknown stored order", () => {
    expect(parseSortOrder("size")).toBe("size");
    expect(parseSortOrder("bogus")).toBe("alpha");
    expect(parseSortOrder(null)).toBe("alpha");
  });
});

describe("groupByState", () => {
  it("orders groups busy, attention, quiet, and keeps row order", () => {
    const groups = groupByState([
      service("e", "evicted"),
      service("s", "stopped"),
      service("i1", "idle"),
      service("f", "failed"),
      service("r", "running"),
      service("i2", "idle"),
      service("d", "disabled", { disabled_reason: "oom" }),
    ]);
    expect(groups.map((g) => g.state)).toEqual([
      "running",
      "failed",
      "disabled",
      "idle",
      "stopped",
      "evicted",
    ]);
    expect(groups[3]?.services.map((s) => s.name)).toEqual(["i1", "i2"]);
  });
});
