import { describe, expect, it } from "vitest";

import { sendsOnEnter, type EnterPress } from "./enterToSend.ts";

const enter: EnterPress = {
  key: "Enter",
  shiftKey: false,
  isComposing: false,
  coarsePointer: false,
};

describe("sendsOnEnter", () => {
  it("sends on a plain Enter", () => {
    expect(sendsOnEnter(enter)).toBe(true);
  });

  it("leaves a newline to Shift+Enter, an IME, and touch", () => {
    expect(sendsOnEnter({ ...enter, shiftKey: true })).toBe(false);
    expect(sendsOnEnter({ ...enter, isComposing: true })).toBe(false);
    expect(sendsOnEnter({ ...enter, coarsePointer: true })).toBe(false);
  });

  it("ignores other keys", () => {
    expect(sendsOnEnter({ ...enter, key: "a" })).toBe(false);
  });
});
