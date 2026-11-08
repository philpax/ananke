// Chat "send" request: kicks off the model if needed, then streams
// the OpenAI-compatible chat completion and folds deltas into the
// chat store snapshot. Chat stays on HTTP rather than the live socket,
// since the OpenAI listener is a separate port.

import { api } from "../../../api/client.ts";
import i18n from "../../../i18n.ts";
import {
  appendDelta,
  buildApiMessages,
  localISO,
  type ApiMessage,
} from "./chatRequest.ts";
import {
  activeRun,
  EMPTY_STATS,
  endRun,
  getSnapshot,
  newMessageId,
  setSnapshot,
  type ChatSnapshot,
  type Run,
  type StreamStats,
} from "./chatStore.ts";
import { readFrame, SseDecoder, usageStats } from "./chatStream.ts";
import { waitForModel } from "./modelStart.ts";

export async function send(
  selectedModel: string,
  openaiBaseUrl: string,
): Promise<void> {
  const snap = getSnapshot();
  const text = snap.input.trim();
  if (!selectedModel || !text || activeRun(snap.chatState)) return;

  // The run takes the composer's text and claims the store before the
  // first await, so a second send, a cancel, or a reset all see it.
  const images = snap.attachments
    .filter((a) => a.type === "image")
    .map((a) => a.content);
  const run: Run = {
    controller: new AbortController(),
    model: selectedModel,
    turn: {
      id: newMessageId(),
      role: "user",
      content: text,
      images: images.length > 0 ? images : undefined,
      timestamp: localISO(),
    },
    draft: { input: snap.input, attachments: snap.attachments },
  };
  const apiMessages = buildApiMessages(
    snap.systemPrompt,
    snap.messages,
    text,
    snap.attachments,
  );
  setSnapshot((prev) => ({
    ...prev,
    messages: [...prev.messages, run.turn],
    attachments: [],
    input: "",
    stats: EMPTY_STATS,
    chatState: { kind: "sending", run },
  }));

  const ctx = runContext(run);
  if (!(await ensureRunning(ctx, selectedModel))) return;
  await stream(ctx, selectedModel, openaiBaseUrl, apiMessages);
}

type RunContext = {
  readonly run: Run;
  /// Applies `updater` only while this send is still the store's run.
  readonly update: (updater: (prev: ChatSnapshot) => ChatSnapshot) => void;
  /// Ends the run with an error, giving the text back if nothing replied.
  readonly fail: (message: string) => void;
  readonly stale: () => boolean;
};

function runContext(run: Run): RunContext {
  const update = (updater: (prev: ChatSnapshot) => ChatSnapshot) =>
    setSnapshot((prev) =>
      activeRun(prev.chatState) === run ? updater(prev) : prev,
    );
  return {
    run,
    update,
    fail: (message) =>
      update((prev) => endRun(prev, run, { kind: "error", message })),
    stale: () => activeRun(getSnapshot().chatState) !== run,
  };
}

/// Starts the model if it is not running. When its state cannot be read,
/// the request goes anyway and the daemon starts the model on demand.
async function ensureRunning(ctx: RunContext, model: string): Promise<boolean> {
  const svc = await api.listServices().then(
    (r) => r.services.find((s) => s.name === model),
    () => undefined,
  );
  if (ctx.stale()) return false;
  if (svc === undefined || svc.state === "running") return true;

  const { run, update } = ctx;
  update((prev) => ({ ...prev, chatState: { kind: "starting", run } }));
  const outcome = await waitForModel(model, run.controller.signal);
  if (ctx.stale()) return false;
  if (outcome.kind === "failed") {
    ctx.fail(
      outcome.failure.kind === "queue_full"
        ? i18n.t("chat.startQueueFull", { model })
        : i18n.t("chat.startFailed", {
            model,
            message: outcome.failure.message,
          }),
    );
    return false;
  }
  update((prev) => ({ ...prev, chatState: { kind: "sending", run } }));
  return outcome.kind === "ready";
}

/// Streams the reply. It counts as complete only on `[DONE]`: a body that
/// just stops, as when the upstream dies mid-reply, is an error.
async function stream(
  ctx: RunContext,
  model: string,
  baseUrl: string,
  messages: readonly ApiMessage[],
): Promise<void> {
  const { run, update, fail } = ctx;
  const startTime = performance.now();
  let stats: StreamStats = EMPTY_STATS;
  const setStats = (next: StreamStats) => {
    stats = next;
    update((prev) => ({ ...prev, stats }));
  };
  const attachStats = () =>
    update((prev) => {
      const last = prev.messages[prev.messages.length - 1];
      if (last?.role !== "assistant") return prev;
      return {
        ...prev,
        messages: [...prev.messages.slice(0, -1), { ...last, stats }],
      };
    });

  try {
    const resp = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: run.controller.signal,
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        stream_options: { include_usage: true },
      }),
    });
    if (!resp.ok) {
      fail(`${resp.status} ${resp.statusText}: ${await resp.text()}`);
      return;
    }
    const reader = resp.body?.getReader();
    if (!reader) {
      fail(i18n.t("chat.streamIncomplete"));
      return;
    }
    update((prev) => ({ ...prev, chatState: { kind: "streaming", run } }));

    const text = new TextDecoder();
    const sse = new SseDecoder();
    let done = false;
    const deliver = (frames: readonly string[]): string | null => {
      for (const data of frames) {
        const frame = readFrame(data);
        if (frame.kind === "done") {
          done = true;
          continue;
        }
        if (frame.kind === "error") return frame.message;
        const { chunk } = frame;
        if (chunk.content || chunk.reasoning) {
          if (stats.ttftMs === null) {
            setStats({ ...stats, ttftMs: performance.now() - startTime });
          }
          update((prev) => ({
            ...prev,
            messages: appendDelta(prev.messages, chunk, () => ({
              id: newMessageId(),
              role: "assistant",
              content: "",
              model,
              timestamp: localISO(),
            })),
          }));
        }
        const usage = usageStats(chunk, (performance.now() - startTime) / 1000);
        if (usage) setStats({ ...usage, ttftMs: stats.ttftMs });
      }
      return null;
    };

    for (;;) {
      const { done: ended, value } = await reader.read();
      const error = ended
        ? deliver(sse.finish())
        : deliver(sse.feed(text.decode(value, { stream: true })));
      if (error !== null) {
        attachStats();
        fail(error);
        return;
      }
      if (ended) break;
    }
    attachStats();
    if (done) {
      update((prev) => ({ ...prev, chatState: { kind: "idle" } }));
    } else {
      fail(i18n.t("chat.streamIncomplete"));
    }
  } catch (e) {
    attachStats();
    if (e instanceof DOMException && e.name === "AbortError") {
      update((prev) => endRun(prev, run, { kind: "idle" }));
    } else {
      fail(e instanceof Error ? e.message : String(e));
    }
  }
}
