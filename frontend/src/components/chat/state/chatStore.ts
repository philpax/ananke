// Chat session store. Holds messages, system prompt, streaming state,
// and input/attachments at module level so they survive navigation
// (component unmount/remount) within the same tab session. The store
// is the single source of truth — ChatView reads from it via
// useSyncExternalStore and mutates it through the exported actions.

import { useSyncExternalStore } from "react";

export type StreamStats = {
  readonly ttftMs: number | null;
  readonly promptTokens: number | null;
  readonly completionTokens: number | null;
  /// Decode throughput: completion tokens divided by the engine-reported
  /// decode window (`timings.predicted_ms`). Null when the engine does
  /// not emit timings, in which case `predictedPerSecond` is the fallback.
  readonly outputTokPerSec: number | null;
  /// Prefill throughput: cache-aware prompt tokens (`timings.prompt_n`)
  /// divided by prefill time (`timings.prompt_ms`). Null when absent.
  readonly inputTokPerSec: number | null;
  /// Effective end-to-end rate: completion tokens over total wall-clock
  /// elapsed. Always computable; shown only as a fallback when the engine
  /// does not provide the input/output split.
  readonly predictedPerSecond: number | null;
};

export type Attachment = {
  readonly name: string;
  readonly size: number;
  readonly type: "text" | "image";
  readonly content: string;
};

export type Message = {
  /// Unique within the tab; timestamps only have second resolution.
  readonly id: number;
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
  readonly reasoning?: string;
  /// The model that wrote an assistant message.
  readonly model?: string;
  readonly images?: readonly string[];
  readonly stats?: StreamStats;
  readonly timestamp: string;
};

/// One send, from the moment it takes the composer's text until it ends.
/// Every write it makes checks that it is still the store's run, so a
/// reset, a cancel, or a later send leaves it nowhere to write.
export type Run = {
  readonly controller: AbortController;
  readonly model: string;
  /// The user message the run added, and the composer it came from, so a
  /// run that ends before any reply can give the text back.
  readonly turn: Message;
  readonly draft: {
    readonly input: string;
    readonly attachments: readonly Attachment[];
  };
};

export type ChatState =
  | { readonly kind: "idle" }
  /// Checking the model, or waiting for the reply's first byte.
  | { readonly kind: "sending"; readonly run: Run }
  | { readonly kind: "starting"; readonly run: Run }
  | { readonly kind: "streaming"; readonly run: Run }
  | { readonly kind: "error"; readonly message: string };

export type ChatSnapshot = {
  readonly messages: readonly Message[];
  readonly systemPrompt: string;
  readonly currentModel: string | null;
  readonly chatState: ChatState;
  readonly stats: StreamStats;
  readonly input: string;
  readonly attachments: readonly Attachment[];
};

export const EMPTY_STATS: StreamStats = {
  ttftMs: null,
  promptTokens: null,
  completionTokens: null,
  outputTokPerSec: null,
  inputTokPerSec: null,
  predictedPerSecond: null,
};

let snapshot: ChatSnapshot = {
  messages: [],
  systemPrompt: "",
  currentModel: null,
  chatState: { kind: "idle" },
  stats: EMPTY_STATS,
  input: "",
  attachments: [],
};

const listeners = new Set<() => void>();

export function setSnapshot(
  updater: (prev: ChatSnapshot) => ChatSnapshot,
): void {
  snapshot = updater(snapshot);
  for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

export function getSnapshot(): ChatSnapshot {
  return snapshot;
}

let nextMessageId = 1;

export function newMessageId(): number {
  return nextMessageId++;
}

/// The run in flight, if any.
export function activeRun(state: ChatState): Run | null {
  return state.kind === "sending" ||
    state.kind === "starting" ||
    state.kind === "streaming"
    ? state.run
    : null;
}

/// Ends `run` in `next`. A run that got no reply takes its message back
/// out and returns its text and attachments to an empty composer, so a
/// retry needs no retyping and the history never holds an unanswered turn.
export function endRun(
  prev: ChatSnapshot,
  run: Run,
  next: ChatState,
): ChatSnapshot {
  const last = prev.messages[prev.messages.length - 1];
  if (last !== run.turn) return { ...prev, chatState: next };
  return {
    ...prev,
    chatState: next,
    messages: prev.messages.slice(0, -1),
    input: prev.input || run.draft.input,
    attachments:
      prev.attachments.length > 0 ? prev.attachments : run.draft.attachments,
  };
}

// --- Actions ---

export function setInput(value: string): void {
  setSnapshot((prev) => ({ ...prev, input: value }));
}

export function saveSystemPrompt(value: string): void {
  setSnapshot((prev) => ({ ...prev, systemPrompt: value }));
  const model = snapshot.currentModel;
  if (model) {
    try {
      localStorage.setItem(`ananke-chat-sys-${model}`, value);
    } catch {
      // localStorage unavailable.
    }
  }
}

/// Switches the model the next message goes to. The conversation carries
/// on: each reply records the model that wrote it, and only the system
/// prompt, which is stored per model, is swapped.
export function selectModel(name: string): void {
  if (name === snapshot.currentModel) return;
  let prompt = "";
  try {
    prompt = localStorage.getItem(`ananke-chat-sys-${name}`) ?? "";
  } catch {
    // localStorage unavailable.
  }
  setSnapshot((prev) => ({
    ...prev,
    systemPrompt: prompt,
    currentModel: name,
  }));
}

export function addAttachment(att: Attachment): void {
  setSnapshot((prev) => ({
    ...prev,
    attachments: [...prev.attachments, att],
  }));
}

export function removeAttachment(index: number): void {
  setSnapshot((prev) => ({
    ...prev,
    attachments: prev.attachments.filter((_, i) => i !== index),
  }));
}

/// Starts a new conversation. The run in flight is aborted and, because
/// the store no longer names it, every write it would still make is
/// dropped.
export function clearConversation(): void {
  activeRun(snapshot.chatState)?.controller.abort();
  setSnapshot((prev) => ({
    ...prev,
    messages: [],
    stats: EMPTY_STATS,
    chatState: { kind: "idle" },
  }));
}

export function cancel(): void {
  const run = activeRun(snapshot.chatState);
  if (!run) return;
  run.controller.abort();
  setSnapshot((prev) => endRun(prev, run, { kind: "idle" }));
}

// --- Send ---

export { send } from "./chatSend.ts";

// --- Hook ---

export function useChat(): ChatSnapshot {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
