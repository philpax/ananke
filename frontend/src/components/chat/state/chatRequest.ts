// Turning the composer and the conversation into a chat completion
// request, and folding the reply's deltas back into the conversation.

import type { Attachment, Message } from "./chatStore.ts";
import type { StreamChunk } from "./chatStream.ts";

type ApiContentPart =
  | { readonly type: "text"; readonly text: string }
  | {
      readonly type: "image_url";
      readonly image_url: { readonly url: string };
    };

export type ApiMessage = {
  readonly role: string;
  readonly content: string | readonly ApiContentPart[];
};

/// The messages for a request: the system prompt, the history, and the
/// new turn, with text attachments inlined and images as parts.
export function buildApiMessages(
  systemPrompt: string,
  history: readonly Message[],
  text: string,
  attachments: readonly Attachment[],
): ApiMessage[] {
  const messages: ApiMessage[] = [];
  if (systemPrompt.trim()) {
    messages.push({ role: "system", content: systemPrompt.trim() });
  }
  for (const m of history) messages.push({ role: m.role, content: m.content });
  let full = text;
  for (const att of attachments) {
    if (att.type === "text") full += `\n\n${att.name}:\n${att.content}`;
  }
  const images = attachments.filter((a) => a.type === "image");
  messages.push({
    role: "user",
    content:
      images.length === 0
        ? full
        : [
            { type: "text", text: full },
            ...images.map((img) => ({
              type: "image_url" as const,
              image_url: { url: img.content },
            })),
          ],
  });
  return messages;
}

/// Appends a chunk's text to the reply, starting the reply on its first
/// text. `start` makes the reply's message when one is needed.
export function appendDelta(
  messages: readonly Message[],
  chunk: StreamChunk,
  start: () => Message,
): readonly Message[] {
  const last = messages[messages.length - 1];
  const reply = last?.role === "assistant" ? last : start();
  const next: Message = {
    ...reply,
    content: reply.content + chunk.content,
    reasoning:
      chunk.reasoning || reply.reasoning
        ? (reply.reasoning ?? "") + chunk.reasoning
        : undefined,
  };
  return reply === last
    ? [...messages.slice(0, -1), next]
    : [...messages, next];
}

/// ISO8601 timestamp in the local timezone (e.g. 2026-06-27T14:30:00+02:00).
export function localISO(d: Date = new Date()): string {
  const off = d.getTimezoneOffset();
  const sign = off <= 0 ? "+" : "-";
  const abs = Math.abs(off);
  const hh = String(Math.floor(abs / 60)).padStart(2, "0");
  const mm = String(abs % 60).padStart(2, "0");
  const local = new Date(d.getTime() - off * 60000).toISOString().slice(0, 19);
  return `${local}${sign}${hh}:${mm}`;
}
