import clsx from "clsx";
import { useEffect, useRef } from "preact/hooks";

import { api } from "../api/client";
import type { Service } from "../api/contract";
import { useResource } from "../api/resource";
import { Button } from "../ui/Button";
import { Copy } from "../ui/Copy";
import { Message } from "../ui/Message";
import { PageHeader } from "../ui/PageHeader";
import { number, time } from "../ui/format";
import { Markdown } from "./Markdown";
import type { ChatTurn } from "./stream";
import type { useChat } from "./useChat";

export function Chat({
  chat,
  services,
  requestedModel,
  active,
}: {
  readonly chat: ReturnType<typeof useChat>;
  readonly services: readonly Service[];
  readonly requestedModel?: string;
  readonly active: boolean;
}) {
  const models = useResource(`models:${active}`, api.models);
  const choices = (models.data?.data ?? []).filter(
    (m) => !m.modality || m.modality === "chat",
  );
  const canSend =
    !chat.generating &&
    !!chat.input.trim() &&
    choices.some((model) => model.id === chat.model);
  const ref = useRef<HTMLDivElement>(null),
    stickRef = useRef(true);
  const setModel = chat.setModel;
  useEffect(() => {
    if (requestedModel) setModel(requestedModel);
  }, [requestedModel, setModel]);
  useEffect(() => {
    if (stickRef.current && ref.current)
      ref.current.scrollTop = ref.current.scrollHeight;
  }, [chat.turns, active]);
  const exchanges = chat.turns.filter((t) => t.role === "user").length;
  const tokenCount = chat.turns.reduce(
    (s, t) => s + (t.inputTokens ?? 0) + (t.outputTokens ?? 0),
    0,
  );
  return (
    <main
      data-screen-label="Chat"
      class="flex h-full min-w-0 flex-1 flex-col bg-canvas"
    >
      <PageHeader
        title="Chat"
        meta={
          <>
            <span class="font-mono text-body">
              {chat.model || "Choose a model"}
            </span>{" "}
            · {exchanges} {exchanges === 1 ? "exchange" : "exchanges"} ·{" "}
            {number(tokenCount, 0)} tokens
          </>
        }
      >
        <Button icon="plus" onClick={chat.clear}>
          New chat
        </Button>
      </PageHeader>
      <div
        ref={ref}
        onScroll={(e) => {
          const el = e.currentTarget;
          stickRef.current =
            el.scrollHeight - el.clientHeight - el.scrollTop < 60;
        }}
        class="
          flex min-h-0 flex-1 flex-col gap-4 overflow-auto px-4 pt-[18px] pb-6
        "
      >
        {chat.turns.map((turn, index) => (
          <Turn key={index} turn={turn} />
        ))}
        {!chat.turns.length && (
          <div
            class="
              flex flex-1 items-center justify-center text-[13px] text-muted
            "
          >
            Choose a model and send a message.
          </div>
        )}
        {chat.error && <Message error={chat.error} />}
      </div>
      <div class="shrink-0 border-t border-line bg-surface">
        <details class="border-b border-line px-4 py-2">
          <summary class="text-[12.5px] font-semibold text-body">
            System prompt
          </summary>
          <textarea
            aria-label="System prompt"
            placeholder="Optional system prompt"
            class="
              mt-2 h-[60px] w-full resize-y border border-line-strong bg-canvas
              px-2.5 py-2 font-mono text-[12.5px] text-ink
            "
            value={chat.system}
            disabled={chat.generating}
            onInput={(e) => chat.setSystem(e.currentTarget.value)}
          />
        </details>
        <div class="flex flex-col gap-2.5 px-4 py-3">
          <div class="flex flex-wrap items-center gap-2.5">
            <select
              aria-label="Chat model"
              disabled={chat.generating}
              class="
                h-8 max-w-full border border-line-strong bg-raised px-2.5
                font-mono text-[12.5px] text-ink scheme-dark
              "
              value={chat.model}
              onChange={(e) => chat.setModel(e.currentTarget.value)}
            >
              <option value="">Choose a model</option>
              {chat.model && !choices.some((m) => m.id === chat.model) && (
                <option value={chat.model}>{chat.model} (unavailable)</option>
              )}
              {choices.map((m) => (
                <option key={m.id} value={m.id}>
                  {services.find((s) => s.name === m.id)?.state === "running"
                    ? "● "
                    : "○ "}
                  {m.id}
                </option>
              ))}
            </select>
            <span class="text-[12.5px] text-muted">
              Enter to send · Shift+Enter for newline
            </span>
          </div>
          {models.error && <Message error={models.error} />}
          <div class="flex items-stretch gap-2">
            <textarea
              aria-label="Message"
              placeholder="Type a message"
              rows={1}
              class="
                min-h-10 min-w-0 flex-1 resize-y border border-line-strong
                bg-canvas px-3 py-2.5 text-[13.5px] text-ink
              "
              value={chat.input}
              onInput={(e) => chat.setInput(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
                  e.preventDefault();
                  if (canSend) {
                    stickRef.current = true;
                    void chat.send();
                  }
                }
              }}
            />
            <Button
              icon={chat.generating ? "stop" : "send"}
              tone="primary"
              class="h-auto px-[18px] text-[13.5px] font-semibold"
              disabled={!chat.generating && !canSend}
              onClick={() => {
                stickRef.current = true;
                if (chat.generating) chat.stop();
                else if (canSend) void chat.send();
              }}
            >
              {chat.generating ? "Stop" : "Send"}
            </Button>
          </div>
        </div>
      </div>
    </main>
  );
}

function Turn({ turn }: { readonly turn: ChatTurn }) {
  const user = turn.role === "user";
  return (
    <article
      class={clsx(
        "w-full border",
        user ? "border-user-line bg-user" : `border-model-line bg-model`,
      )}
    >
      <div
        class={clsx(
          `flex flex-wrap items-center gap-2.5 border-b px-3 py-2`,
          user
            ? `border-user-line-soft bg-hover`
            : `border-model-line bg-model-raised`,
        )}
      >
        <h2
          class={clsx(
            `text-[12.5px] text-ink`,
            user
              ? `font-semibold tracking-[0.06em] uppercase`
              : `font-mono font-bold`,
          )}
        >
          {user ? "You" : turn.model}
        </h2>
        <time
          class="font-mono text-xs text-muted"
          dateTime={new Date(turn.at).toISOString()}
        >
          {time(turn.at)}
        </time>
        {!user && (
          <span class="font-mono text-xs text-body">
            {number(turn.inputTokens, 0)} in · {number(turn.outputTokens, 0)}{" "}
            out · {number(turn.speed)} tok/s
          </span>
        )}
        <span class="ml-auto flex">
          <Copy
            text={turn.content}
            label={user ? "your message" : "model reply"}
          />
        </span>
      </div>
      <div class="p-3">
        {turn.reasoning && (
          <details class="mb-3 border border-line bg-raised">
            <summary class="px-2.5 py-[7px] text-[12.5px] text-body">
              Reasoning · {turn.reasoning.split("\n").length} lines
            </summary>
            <div
              class="
                px-2.5 pb-2.5 text-[13px] leading-[1.65] whitespace-pre-wrap
                text-body
              "
            >
              {turn.reasoning}
            </div>
          </details>
        )}
        {user ? (
          <p class="text-[14.5px] leading-[1.6] whitespace-pre-wrap text-ink">
            {turn.content}
          </p>
        ) : (
          <Markdown text={turn.content} />
        )}
      </div>
    </article>
  );
}
