// Figures derived from the conversation for the header and each reply.

import type { Message, StreamStats } from "./state/chatStore.ts";

export type ConversationMeta = {
  readonly exchanges: number;
  /// The tokens the last reply's request held: its prompt, which carries
  /// the whole history, and its output. `null` before any usage arrives.
  readonly contextTokens: number | null;
};

export function conversationMeta(
  messages: readonly Message[],
): ConversationMeta {
  const exchanges = messages.filter((m) => m.role === "user").length;
  const last = messages.findLast(
    (m) => m.role === "assistant" && m.stats?.promptTokens != null,
  );
  const stats = last?.stats;
  return {
    exchanges,
    contextTokens:
      stats?.promptTokens != null
        ? stats.promptTokens + (stats.completionTokens ?? 0)
        : null,
  };
}

export type ReplyRates =
  | { readonly kind: "split"; readonly input: number; readonly output: number }
  | { readonly kind: "effective"; readonly rate: number }
  | { readonly kind: "none" };

/// The engine's input and output rates when it reports both, otherwise
/// the effective end-to-end rate.
export function replyRates(stats: StreamStats): ReplyRates {
  if (stats.inputTokPerSec !== null && stats.outputTokPerSec !== null) {
    return {
      kind: "split",
      input: stats.inputTokPerSec,
      output: stats.outputTokPerSec,
    };
  }
  if (stats.predictedPerSecond !== null) {
    return { kind: "effective", rate: stats.predictedPerSecond };
  }
  return { kind: "none" };
}
