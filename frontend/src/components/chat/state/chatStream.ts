// Reading an OpenAI-compatible chat completion stream: server-sent event
// frames, each a JSON chunk, ending with `[DONE]`. The chunk shape is the
// upstream engine's, not a daemon type, so it is validated here rather
// than generated.

import type { StreamStats } from "./chatStore.ts";

/// Splits a byte-decoded SSE body into the `data` of each event. Events
/// end at a blank line; a multi-line `data` field is rejoined with `\n`.
export class SseDecoder {
  private buffer = "";

  feed(text: string): string[] {
    this.buffer += text;
    const frames: string[] = [];
    for (;;) {
      const separator = /\r?\n\r?\n/.exec(this.buffer);
      if (!separator) break;
      const event = this.buffer.slice(0, separator.index);
      this.buffer = this.buffer.slice(separator.index + separator[0].length);
      const data = event
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) frames.push(data);
    }
    return frames;
  }

  /// Flushes a final event the body ended without terminating.
  finish(): string[] {
    return this.feed("\n\n");
  }
}

export type StreamChunk = {
  readonly content: string;
  readonly reasoning: string;
  readonly usage: {
    readonly promptTokens: number | null;
    readonly completionTokens: number | null;
  } | null;
  readonly timings: {
    readonly promptMs: number | null;
    readonly predictedMs: number | null;
    readonly promptN: number | null;
  } | null;
};

export type StreamFrame =
  | { readonly kind: "done" }
  | { readonly kind: "chunk"; readonly chunk: StreamChunk }
  | { readonly kind: "error"; readonly message: string };

/// Reads one event's data. Frames that are not JSON objects are errors:
/// a stream that cannot be read cannot be trusted to be complete.
export function readFrame(data: string): StreamFrame {
  if (data === "[DONE]") return { kind: "done" };
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return { kind: "error", message: `unreadable stream frame: ${data}` };
  }
  if (!isRecord(parsed)) {
    return { kind: "error", message: `unreadable stream frame: ${data}` };
  }
  if (parsed.error !== undefined) {
    return { kind: "error", message: errorText(parsed.error) };
  }
  const choice = Array.isArray(parsed.choices) ? parsed.choices[0] : undefined;
  const delta = isRecord(choice) && isRecord(choice.delta) ? choice.delta : {};
  const usage = isRecord(parsed.usage) ? parsed.usage : null;
  const timings = isRecord(parsed.timings) ? parsed.timings : null;
  return {
    kind: "chunk",
    chunk: {
      content: str(delta.content),
      // llama.cpp sends `reasoning_content`; some engines `reasoning`.
      reasoning: str(delta.reasoning_content) || str(delta.reasoning),
      usage: usage && {
        promptTokens: num(usage.prompt_tokens),
        completionTokens: num(usage.completion_tokens),
      },
      timings: timings && {
        promptMs: num(timings.prompt_ms),
        predictedMs: num(timings.predicted_ms),
        promptN: num(timings.prompt_n),
      },
    },
  };
}

/// The stats a usage chunk yields, `elapsedSec` after the request began.
///
/// llama.cpp's `timings` sit beside `usage` in the final chunk: prefill
/// and decode times, and `prompt_n`, the prompt tokens actually evaluated
/// (excluding the cache, unlike `usage.prompt_tokens`). They give the true
/// input and output rates; without them only the effective rate is known.
export function usageStats(
  chunk: StreamChunk,
  elapsedSec: number,
): Omit<StreamStats, "ttftMs"> | null {
  if (!chunk.usage) return null;
  const completion = chunk.usage.completionTokens ?? 0;
  const t = chunk.timings;
  return {
    promptTokens: chunk.usage.promptTokens,
    completionTokens: completion,
    predictedPerSecond: elapsedSec > 0 ? completion / elapsedSec : null,
    inputTokPerSec:
      t?.promptMs && t.promptN ? t.promptN / (t.promptMs / 1000) : null,
    outputTokPerSec:
      t?.predictedMs && completion > 0
        ? completion / (t.predictedMs / 1000)
        : null,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function errorText(error: unknown): string {
  if (typeof error === "string") return error;
  if (isRecord(error) && typeof error.message === "string") {
    return error.message;
  }
  return JSON.stringify(error);
}
