import { describe, expect, it } from "vitest";

import { readFrame, SseDecoder, usageStats } from "./chatStream.ts";

describe("SseDecoder", () => {
  it("joins events split across reads", () => {
    const sse = new SseDecoder();
    expect(sse.feed('data: {"a"')).toEqual([]);
    expect(sse.feed(":1}\n\ndata: [DO")).toEqual(['{"a":1}']);
    expect(sse.feed("NE]\r\n\r\n")).toEqual(["[DONE]"]);
  });

  it("rejoins multi-line data and skips comments", () => {
    const sse = new SseDecoder();
    expect(sse.feed(": keep-alive\n\ndata: a\ndata: b\n\n")).toEqual(["a\nb"]);
  });

  it("flushes an unterminated final event", () => {
    const sse = new SseDecoder();
    expect(sse.feed("data: [DONE]")).toEqual([]);
    expect(sse.finish()).toEqual(["[DONE]"]);
  });
});

describe("readFrame", () => {
  it("reads content and either reasoning field", () => {
    const frame = readFrame(
      JSON.stringify({ choices: [{ delta: { reasoning: "hm" } }] }),
    );
    expect(frame).toMatchObject({
      kind: "chunk",
      chunk: { content: "", reasoning: "hm", usage: null },
    });
  });

  it("reports an error frame and an unreadable one", () => {
    expect(readFrame('{"error":{"message":"context full"}}')).toEqual({
      kind: "error",
      message: "context full",
    });
    expect(readFrame("{nope").kind).toBe("error");
  });
});

describe("usageStats", () => {
  const chunk = (timings: boolean) => {
    const frame = readFrame(
      JSON.stringify({
        choices: [],
        usage: { prompt_tokens: 100, completion_tokens: 50 },
        ...(timings
          ? { timings: { prompt_ms: 500, predicted_ms: 2000, prompt_n: 40 } }
          : {}),
      }),
    );
    if (frame.kind !== "chunk") throw new Error("not a chunk");
    return frame.chunk;
  };

  it("splits input and output rates from llama.cpp timings", () => {
    expect(usageStats(chunk(true), 5)).toEqual({
      promptTokens: 100,
      completionTokens: 50,
      predictedPerSecond: 10,
      inputTokPerSec: 80,
      outputTokPerSec: 25,
    });
  });

  it("falls back to the effective rate without timings", () => {
    expect(usageStats(chunk(false), 5)).toMatchObject({
      predictedPerSecond: 10,
      inputTokPerSec: null,
      outputTokPerSec: null,
    });
  });
});
