import { expect, test } from "vitest";

import { SseDecoder } from "./stream";

test("stream framing survives every split point, CRLF, comments, multiline data, and final unterminated frames", () => {
  const text = ": keepalive\r\ndata: first\r\ndata: second\r\n\r\ndata: [DONE]";
  for (let split = 0; split <= text.length; split++) {
    const parser = new SseDecoder();
    expect([
      ...parser.feed(text.slice(0, split)),
      ...parser.feed(text.slice(split)),
      ...parser.finish(),
    ]).toEqual(["first\nsecond", "[DONE]"]);
  }
});
