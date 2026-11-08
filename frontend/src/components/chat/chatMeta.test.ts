import { describe, expect, it } from "vitest";

import { conversationMeta, replyRates } from "./chatMeta.ts";
import { EMPTY_STATS, type Message } from "./state/chatStore.ts";

function message(
  id: number,
  role: Message["role"],
  promptTokens: number | null = null,
): Message {
  return {
    id,
    role,
    content: "",
    timestamp: "t",
    stats:
      promptTokens === null
        ? undefined
        : { ...EMPTY_STATS, promptTokens, completionTokens: 10 },
  };
}

describe("conversationMeta", () => {
  it("counts exchanges and the last reply's context", () => {
    expect(
      conversationMeta([
        message(1, "user"),
        message(2, "assistant", 20),
        message(3, "user"),
        message(4, "assistant", 50),
        message(5, "user"),
      ]),
    ).toEqual({ exchanges: 3, contextTokens: 60 });
  });

  it("has no context before any usage", () => {
    expect(conversationMeta([message(1, "user")])).toEqual({
      exchanges: 1,
      contextTokens: null,
    });
  });
});

describe("replyRates", () => {
  it("prefers the engine's split, then the effective rate", () => {
    expect(
      replyRates({
        ...EMPTY_STATS,
        inputTokPerSec: 800,
        outputTokPerSec: 40,
        predictedPerSecond: 30,
      }),
    ).toEqual({ kind: "split", input: 800, output: 40 });
    expect(replyRates({ ...EMPTY_STATS, predictedPerSecond: 30 })).toEqual({
      kind: "effective",
      rate: 30,
    });
    expect(replyRates(EMPTY_STATS)).toEqual({ kind: "none" });
  });
});
