import { expect, test } from "vitest";

import { chunkCodec, codec, eventCodec } from "./contract";

test("generated wire validators reject malformed inventory and distinguish stream variants", () => {
  expect(
    codec("ServicesResponse")({
      services: [{ name: "model" }],
      openai_api_port: 7070,
    }),
  ).toBe(false);
  expect(
    eventCodec({
      type: "state_changed",
      service: "model",
      from: "idle",
      to: "running",
      at_ms: 100,
    }),
  ).toBe(true);
  expect(
    eventCodec({ type: "state_changed", service: "model", to: "running" }),
  ).toBe(false);
  expect(
    chunkCodec({
      choices: [{ delta: { content: "Hello", reasoning_content: "Think" } }],
      usage: { completion_tokens: 4 },
    }),
  ).toBe(true);
});
