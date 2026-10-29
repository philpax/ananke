import { describe, expect, it } from "vitest";

import { sectionForPath, sectionHref, serviceFromPath } from "./navigation.ts";

describe("sectionForPath", () => {
  it.each([
    ["/", "services"],
    ["/services/a", "services"],
    ["/services/a%2Fb", "services"],
    ["/services/%E0", "services"],
    ["/chat", "chat"],
    ["/config/", "config"],
    ["/events", "events"],
    ["/stats", "stats"],
    ["/reservations", "reservations"],
    ["/nope", undefined],
    ["/servicesx", undefined],
  ])("%s is %s", (path, section) => {
    expect(sectionForPath(path)).toBe(section);
  });
});

describe("serviceFromPath", () => {
  it.each([
    ["/services/a", "a"],
    ["/services/a%2Fb", "a/b"],
    ["/services/%E0", undefined],
    ["/services/", undefined],
    ["/", undefined],
    ["/chat", undefined],
  ])("%s names %s", (path, service) => {
    expect(serviceFromPath(path)).toBe(service);
  });
});

describe("sectionHref", () => {
  it("opens the list without a last service", () => {
    expect(sectionHref("services", undefined)).toBe("/");
  });

  it("reopens the last service, escaped", () => {
    expect(sectionHref("services", { name: "a/b", openai_compat: true })).toBe(
      "/services/a%2Fb",
    );
  });

  it("preselects a chat model in chat", () => {
    expect(sectionHref("chat", { name: "qwen", openai_compat: true })).toBe(
      "/chat?model=qwen",
    );
  });

  it("does not preselect an embedding model in chat", () => {
    expect(
      sectionHref("chat", {
        name: "bge",
        modality: "embedding",
        openai_compat: true,
      }),
    ).toBe("/chat");
  });

  it("does not preselect a service outside the OpenAI API in chat", () => {
    expect(sectionHref("chat", { name: "comfy", openai_compat: false })).toBe(
      "/chat",
    );
  });

  it("carries the time range into Services and Stats only", () => {
    const qwen = { name: "qwen", openai_compat: true };
    expect(sectionHref("services", undefined, "?range=6h")).toBe("/?range=6h");
    expect(sectionHref("services", qwen, "?range=6h")).toBe(
      "/services/qwen?range=6h",
    );
    expect(sectionHref("stats", qwen, "?from=1&to=2")).toBe(
      "/stats?from=1&to=2",
    );
    expect(sectionHref("chat", qwen, "?range=6h")).toBe("/chat?model=qwen");
    expect(sectionHref("events", qwen, "?range=6h")).toBe("/events?range=6h");
    expect(sectionHref("config", qwen, "?range=6h")).toBe("/config");
  });

  it("leads back to the list from within Services", () => {
    const qwen = { name: "qwen", openai_compat: true };
    expect(sectionHref("services", qwen, "?range=1h", "services")).toBe(
      "/?range=1h",
    );
    expect(sectionHref("services", qwen, "", "stats")).toBe("/services/qwen");
  });

  it("ignores the last service elsewhere", () => {
    expect(sectionHref("config", { name: "qwen", openai_compat: true })).toBe(
      "/config",
    );
  });
});
