// Chat interface — a web equivalent of `anankectl chat`. The operator
// picks a model, enters a system prompt, and chats with streaming
// responses, token stats, and file attachments.
//
// Chat state (messages, system prompt, input, attachments, streaming
// state) lives in the module-level chatStore so it survives navigation
// away and back within the same tab session.

import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

import { useServices, useInfo } from "../../api/hooks.ts";
import { type ServiceSummary } from "../../api/client.ts";
import { openaiBaseUrlFromListen } from "../../util.ts";
import {
  addAttachment,
  cancel as cancelSend,
  clearConversation,
  removeAttachment,
  saveSystemPrompt,
  selectModel,
  send,
  setInput,
  useChat,
  type Message,
  type StreamStats,
} from "../../api/chatStore.ts";
import { Spinner } from "../ui/Spinner.tsx";
import { Button } from "../ui/Button.tsx";
import { ButtonLink } from "../ui/ButtonLink.tsx";
import { Badge } from "../ui/Badge.tsx";
import { EmptyState } from "../ui/EmptyState.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { CopyButton } from "../ui/CopyButton.tsx";
import { ViewHeader } from "../ui/ViewHeader.tsx";
import { ExternalLinkIcon, TrashIcon } from "../ui/icons.tsx";

export function ChatView() {
  const { t } = useTranslation();
  const services = useServices();
  const info = useInfo();
  const [searchParams, setSearchParams] = useSearchParams();
  const chat = useChat();

  const chatModels = (services.data ?? [])
    .filter(
      (s) =>
        s.modality !== "embedding" && !s.name.toLowerCase().includes("comfyui"),
    )
    .sort((a, b) => {
      const ar = a.state === "running" ? 0 : 1;
      const br = b.state === "running" ? 0 : 1;
      if (ar !== br) return ar - br;
      return a.name.localeCompare(b.name);
    });
  const paramModel = searchParams.get("model");

  // Sync URL → store, but only when the URL explicitly specifies a
  // model AND there is no active conversation. The store is the source
  // of truth for session persistence; navigating to /chat without
  // ?model= should not wipe the session, and navigating from a service
  // detail view with ?model=foo should not clobber an existing
  // conversation with a different model.
  useEffect(() => {
    if (
      paramModel &&
      paramModel !== chat.currentModel &&
      chat.messages.length === 0
    ) {
      selectModel(paramModel);
    }
  }, [paramModel, chat.currentModel, chat.messages.length]);

  // The store's currentModel is the effective selection — it survives
  // navigation away and back even when the URL lacks ?model=.
  const selectedModel = chat.currentModel;

  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    autoScrollRef.current = atBottom;
  }

  useEffect(() => {
    if (scrollRef.current && autoScrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chat.messages]);

  async function handleSend() {
    if (!selectedModel) return;
    const baseUrl = openaiBaseUrlFromListen(
      info.data?.openai_listen ?? "0.0.0.0:7070",
    );
    await send(selectedModel, baseUrl);
  }

  function handleSelectModel(name: string) {
    setSearchParams({ model: name });
  }

  async function handleFiles(files: FileList) {
    const svc = selectedModel
      ? (services.data ?? []).find((s) => s.name === selectedModel)
      : null;
    const hasVision = svc?.has_mmproj ?? false;

    for (const file of Array.from(files)) {
      if (file.type.startsWith("image/")) {
        if (!hasVision) continue;
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result;
          if (typeof result === "string") {
            addAttachment({
              name: file.name,
              size: file.size,
              type: "image",
              content: result,
            });
          }
        };
        reader.readAsDataURL(file);
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result;
          if (typeof result === "string") {
            addAttachment({
              name: file.name,
              size: file.size,
              type: "text",
              content: result,
            });
          }
        };
        reader.readAsText(file);
      }
    }
  }

  if (services.isPending) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  const inputDisabled = !selectedModel || chat.chatState.kind === "starting";

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <ViewHeader>
        <h1 className="eyebrow !text-primary">{t("chat.title")}</h1>
      </ViewHeader>

      {/* Messages */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-auto px-4 py-4"
      >
        {chat.messages.length === 0 ? (
          <>
            <EmptyState message={t("chat.emptyState")} />
            {chat.chatState.kind === "starting" && (
              <div className="flex items-center justify-center gap-2 py-2 text-sm text-tertiary">
                <Spinner />
                {t("chat.starting", { model: selectedModel })}
              </div>
            )}
          </>
        ) : (
          chat.messages.map((msg, i) => (
            <MessageBubble
              key={msg.timestamp}
              message={msg}
              modelName={selectedModel}
              liveStats={
                chat.chatState.kind === "streaming" &&
                i === chat.messages.length - 1 &&
                msg.role === "assistant"
                  ? chat.stats
                  : null
              }
            />
          ))
        )}
        {chat.messages.length > 0 && chat.chatState.kind === "starting" && (
          <div className="flex items-center gap-2 py-2 text-sm text-tertiary">
            <Spinner />
            {t("chat.starting", { model: selectedModel })}
          </div>
        )}
        {chat.chatState.kind === "error" && (
          <div className="mt-2 rounded-sm border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            {chat.chatState.message}
          </div>
        )}
      </div>

      {/* Composer block: sits at the bottom of the flex column, below
          the flex-1 message list, so it never scrolls with the
          conversation on mobile or desktop. */}
      <div className="shrink-0 bg-surface">
        {/* System prompt */}
        {selectedModel && (
          <details className="shrink-0 border-t border-border-default px-4 py-2">
            <summary className="eyebrow cursor-pointer select-none hover:text-secondary">
              {t("chat.systemPrompt")}
            </summary>
            <textarea
              value={chat.systemPrompt}
              onChange={(e) => saveSystemPrompt(e.target.value)}
              placeholder={t("chat.systemPromptPlaceholder")}
              className="mt-1 h-20 w-full resize-none rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary placeholder:text-tertiary focus:border-accent focus:outline-none"
            />
          </details>
        )}

        {/* Attachments preview */}
        {chat.attachments.length > 0 && (
          <div className="shrink-0 flex flex-wrap items-center gap-2 border-t border-border-default px-4 py-2">
            {chat.attachments.map((att, i) => (
              <div
                key={i}
                className="flex items-center gap-1 rounded-sm bg-elevated px-2 py-0.5 text-xs text-secondary"
              >
                {att.type === "image" && (
                  <img
                    src={att.content}
                    alt={att.name}
                    className="h-6 w-6 rounded object-cover"
                  />
                )}
                <span>{att.name}</span>
                <button
                  onClick={() => removeAttachment(i)}
                  className="text-tertiary hover:text-danger"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Composer: model picker + stats sit next to the input, so the
          controls you use most are all within reach of the textbox. */}
        <div className="shrink-0 border-t border-border-default px-4 py-3">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <ModelDropdown
              models={chatModels}
              selected={selectedModel}
              onSelect={handleSelectModel}
              className="min-w-0 flex-1"
            />
            {selectedModel && (
              <CopyButton
                value={selectedModel}
                className="h-7 rounded-md bg-elevated px-2 text-xs font-medium text-primary hover:bg-border-strong"
              />
            )}
            {selectedModel && (
              <ButtonLink
                to={`/services/${encodeURIComponent(selectedModel)}`}
                variant="secondary"
                size="sm"
                className="w-7 justify-center px-0"
              >
                <ExternalLinkIcon />
              </ButtonLink>
            )}
            {selectedModel && (
              <Button
                variant="secondary"
                size="sm"
                className="w-7 justify-center px-0"
                onClick={clearConversation}
              >
                <TrashIcon />
              </Button>
            )}
          </div>
          <div className="flex items-stretch gap-2">
            <textarea
              value={chat.input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  // On touch devices, Enter inserts a newline instead of
                  // sending. Coarse-pointer devices are phones/tablets.
                  if (window.matchMedia("(pointer: coarse)").matches) return;
                  e.preventDefault();
                  void handleSend();
                }
              }}
              placeholder={
                selectedModel
                  ? chat.chatState.kind === "starting"
                    ? t("chat.startingModelPlaceholder")
                    : t("chat.typePlaceholder")
                  : t("chat.selectModelFirst")
              }
              disabled={inputDisabled}
              rows={3}
              className="flex-1 resize-none overflow-y-auto rounded-sm border border-border-default bg-base px-2 py-2.5 text-base text-primary placeholder:text-tertiary focus:border-accent focus:outline-none disabled:opacity-50 md:text-sm"
            />
            <label className="flex w-10 shrink-0 self-stretch cursor-pointer items-center justify-center rounded-md bg-elevated text-lg text-secondary hover:bg-border-strong">
              <input
                type="file"
                multiple
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) handleFiles(e.target.files);
                  e.target.value = "";
                }}
              />
              +
            </label>
            {chat.chatState.kind === "streaming" ||
            chat.chatState.kind === "starting" ? (
              <button
                onClick={cancelSend}
                disabled={chat.chatState.kind === "starting"}
                className="shrink-0 self-stretch rounded-md bg-danger px-3 text-sm font-medium text-white hover:bg-danger/90 disabled:opacity-40"
              >
                {t("chat.stop")}
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!selectedModel || !chat.input.trim()}
                className="inline-flex shrink-0 self-stretch items-center justify-center gap-1.5 rounded-md bg-action-iris px-3 text-sm font-medium text-white transition-[filter] hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {t("chat.send")}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ModelDropdown({
  models,
  selected,
  onSelect,
  className = "",
}: {
  models: ServiceSummary[];
  selected: string | null;
  onSelect: (name: string) => void;
  className?: string;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent | TouchEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        close();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
    }
  }, [open]);

  function close() {
    setOpen(false);
    setFilter("");
  }

  const selectedSvc = models.find((s) => s.name === selected);
  const filtered = filter
    ? models.filter((s) => s.name.toLowerCase().includes(filter.toLowerCase()))
    : models;

  return (
    <div ref={ref} className={`relative ${className}`}>
      {open ? (
        <input
          ref={inputRef}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              close();
            } else if (e.key === "Enter" && filtered.length > 0) {
              onSelect(filtered[0].name);
              close();
            }
          }}
          placeholder={t("chat.filterModels")}
          className="h-7 w-full rounded-sm border border-border-default bg-surface px-2 text-sm text-primary placeholder:text-tertiary focus:border-accent focus:outline-none"
        />
      ) : (
        <button
          onClick={() => setOpen(true)}
          className="flex h-7 w-full min-w-0 items-center gap-2 rounded-sm border border-border-default bg-surface px-2 text-sm text-primary hover:bg-elevated"
        >
          {selectedSvc ? (
            <>
              <StatusDot state={selectedSvc.state} />
              <span className="min-w-0 truncate font-mono">
                {selectedSvc.name}
              </span>
              {selectedSvc.has_mmproj && (
                <Badge variant="vision" className="shrink-0">
                  vision
                </Badge>
              )}
            </>
          ) : (
            <span className="text-tertiary">
              {t("chat.selectModelPlaceholder")}
            </span>
          )}
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="ml-auto shrink-0 text-tertiary"
          >
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
      )}
      {open && (
        <div className="absolute bottom-full left-0 z-20 mb-1 max-h-72 w-full overflow-auto rounded-md border border-border-default bg-surface shadow-lg">
          {filtered.length === 0 ? (
            <div className="px-3 py-2 text-sm text-tertiary">
              {t("chat.noMatchingModels")}
            </div>
          ) : (
            filtered.map((s) => (
              <button
                key={s.name}
                onClick={() => {
                  onSelect(s.name);
                  close();
                }}
                className={`flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-elevated ${
                  s.name === selected ? "bg-elevated" : ""
                }`}
              >
                <StatusDot state={s.state} />
                <span className="font-mono text-sm text-primary">{s.name}</span>
                {s.has_mmproj && <Badge variant="vision">vision</Badge>}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function MessageBubble({
  message,
  modelName,
  liveStats,
}: {
  message: Message;
  modelName: string | null;
  liveStats: StreamStats | null;
}) {
  const { t } = useTranslation();
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isAssistant = message.role === "assistant";

  const label = isAssistant ? (modelName ?? t("chat.assistant")) : message.role;
  const displayStats = liveStats ?? message.stats ?? null;

  return (
    <div className={`mb-4 ${isSystem ? "opacity-60" : ""}`}>
      <div className="mb-1 flex flex-wrap items-center gap-x-3 gap-y-0">
        <span className="eyebrow min-w-0 shrink truncate">{label}</span>
        <span className="shrink-0 font-mono text-xs text-tertiary">
          {message.timestamp}
        </span>
        {/* Forces the stats onto their own row on mobile, where the
            label/timestamp/stats combination is too wide to fit on
            one line; hidden on larger screens so it never forces a
            break there. */}
        <span className="basis-full sm:hidden" />
        {isAssistant && displayStats && displayStats.promptTokens !== null && (
          <span className="flex items-center gap-3 text-xs text-tertiary">
            <span>
              {t("chat.promptTokens", { value: displayStats.promptTokens })}
            </span>
            {displayStats.completionTokens !== null && (
              <span>
                {t("chat.outputTokens", {
                  value: displayStats.completionTokens,
                })}
              </span>
            )}
            {displayStats.predictedPerSecond !== null && (
              <span>
                {t("chat.tokensPerSecond", {
                  value: displayStats.predictedPerSecond.toFixed(1),
                })}
              </span>
            )}
          </span>
        )}
      </div>
      <div
        className={`rounded-md px-4 py-3 text-sm ring-1 ring-inset ${
          isUser
            ? "bg-accent/10 ring-accent/20"
            : "bg-elevated ring-border-default/60"
        } ${isSystem ? "text-secondary" : "text-primary"}`}
      >
        {isAssistant && message.reasoning && (
          <details
            open
            className="open:mb-2 open:border-b open:border-border-default open:pb-2 [&_summary]:list-none"
          >
            <summary className="cursor-pointer select-none text-xs text-secondary hover:text-primary">
              {t("chat.reasoning")}
            </summary>
            <div className="mt-1 max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-xs text-secondary">
              {message.reasoning}
            </div>
          </details>
        )}
        {isAssistant ? (
          message.content ? (
            <div className="flex flex-col gap-3 overflow-hidden break-words">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
                components={{
                  code: ({ children, className }) => {
                    const lang = className?.replace("language-", "");
                    return (
                      <pre className="my-2 overflow-x-auto bg-base p-2 font-mono text-xs">
                        <code data-lang={lang}>{children}</code>
                      </pre>
                    );
                  },
                  table: ({ children }) => (
                    <div className="my-2 overflow-x-auto">
                      <table className="min-w-full border-collapse text-xs">
                        {children}
                      </table>
                    </div>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          ) : null
        ) : (
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
        )}
        {message.images && message.images.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {message.images.map((src, i) => (
              <img
                key={i}
                src={src}
                alt={`attachment ${i + 1}`}
                className="max-h-40 object-cover"
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
