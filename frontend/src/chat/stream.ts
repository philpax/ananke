import { command } from "../api/socket";
import type { Result } from "../api/client";
import { chunkCodec } from "../api/contract";
import type { Schemas } from "../api/contract";

export type ChatTurn = {
  readonly role: "user" | "assistant";
  readonly content: string;
  readonly reasoning: string;
  readonly model: string;
  readonly at: number;
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly speed?: number;
};
export type ChatRequest = {
  readonly model: string;
  readonly system: string;
  readonly turns: readonly ChatTurn[];
};

export class SseDecoder {
  private buffer = "";
  feed(text: string): string[] {
    this.buffer += text;
    const frames: string[] = [];
    for (;;) {
      const separator = /\r?\n\r?\n/.exec(this.buffer);
      if (!separator || separator.index === undefined) break;
      const frame = this.buffer.slice(0, separator.index);
      this.buffer = this.buffer.slice(separator.index + separator[0].length);
      const data = frame
        .split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) frames.push(data);
    }
    return frames;
  }
  finish(): string[] {
    return this.feed("\n\n");
  }
}

export async function streamChat(
  request: ChatRequest,
  signal: AbortSignal,
  onChunk: (chunk: Schemas["ChatCompletionChunk"]) => void,
): Promise<Result<void>> {
  try {
    const body: Schemas["ChatCompletionEnvelope"] = {
      model: request.model,
      messages: [
        ...(request.system
          ? [{ role: "system", content: request.system }]
          : []),
        ...request.turns.map((t) => ({ role: t.role, content: t.content })),
      ],
      stream: true,
      stream_options: { include_usage: true },
    };
    const sse = new SseDecoder();
    let doneMarker = false;
    function deliver(frames: readonly string[]) {
      for (const data of frames) {
        if (data === "[DONE]") {
          doneMarker = true;
          continue;
        }
        const chunk: unknown = JSON.parse(data);
        if (chunk && typeof chunk === "object" && "error" in chunk)
          throw new Error(`Chat: ${JSON.stringify(chunk.error)}`);
        if (!chunkCodec(chunk))
          throw new Error("Chat: the server returns an invalid stream frame.");
        onChunk(chunk);
      }
    }
    const result = await command({ type: "chat", body }, signal, (text) =>
      deliver(sse.feed(text)),
    );
    if (result.kind === "error")
      return signal.aborted
        ? { kind: "error", message: "Generation stopped." }
        : result;
    deliver(sse.finish());
    return doneMarker
      ? { kind: "ok", value: undefined }
      : {
          kind: "error",
          message: "Chat: the response stream ends before completion.",
        };
  } catch (error) {
    return {
      kind: "error",
      message: signal.aborted
        ? "Generation stopped."
        : error instanceof Error
          ? error.message
          : String(error),
    };
  }
}
