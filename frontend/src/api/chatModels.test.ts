import { describe, expect, it } from "vitest";

import { isChatModel } from "./chatModels.ts";

describe("isChatModel", () => {
  it("needs the OpenAI API and a chat modality", () => {
    expect(isChatModel({ openai_compat: true })).toBe(true);
    expect(isChatModel({ openai_compat: true, modality: "chat" })).toBe(true);
    expect(isChatModel({ openai_compat: true, modality: "embedding" })).toBe(
      false,
    );
    expect(isChatModel({ openai_compat: false })).toBe(false);
  });
});
