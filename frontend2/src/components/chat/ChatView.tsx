// Chat interface — a web equivalent of `anankectl chat`. The operator
// picks a model, enters a system prompt, and chats with streaming
// responses, token stats, and file attachments.

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DOMPurify from "dompurify";

import { useServices, useInfo } from "../../api/hooks.ts";
import { api, type ServiceSummary } from "../../api/client.ts";
import { openaiBaseUrlFromListen } from "../../util.ts";
import { Spinner } from "../ui/Spinner.tsx";
import { Button } from "../ui/Button.tsx";
import { ButtonLink } from "../ui/ButtonLink.tsx";
import { Badge } from "../ui/Badge.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { CopyButton } from "../ui/CopyButton.tsx";

type Message = {
  role: "system" | "user" | "assistant";
  content: string;
  reasoning?: string;
  images?: string[];
  stats?: StreamStats;
};

type ChatState =
  | { kind: "idle" }
  | { kind: "starting" }
  | { kind: "streaming"; controller: AbortController }
  | { kind: "error"; message: string };

type StreamStats = {
  ttftMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  predictedPerSecond: number | null;
};

type Attachment = {
  name: string;
  size: number;
  type: "text" | "image";
  content: string;
};

export function ChatView() {
  const services = useServices();
  const info = useInfo();
  const [searchParams, setSearchParams] = useSearchParams();

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
  const selectedModel = paramModel ?? null;

  function selectModel(name: string) {
    try {
      const saved = localStorage.getItem(`ananke-chat-sys-${name}`);
      setSystemPrompt(saved ?? "");
    } catch {
      // localStorage unavailable.
    }
    setMessages([]);
    setStats({
      ttftMs: null,
      promptTokens: null,
      completionTokens: null,
      predictedPerSecond: null,
    });
    setChatState({ kind: "idle" });
    setSearchParams({ model: name });
  }

  const [systemPrompt, setSystemPrompt] = useState(() => {
    if (paramModel) {
      try {
        return localStorage.getItem(`ananke-chat-sys-${paramModel}`) ?? "";
      } catch {
        // localStorage unavailable.
      }
    }
    return "";
  });
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [chatState, setChatState] = useState<ChatState>({ kind: "idle" });
  const [stats, setStats] = useState<StreamStats>({
    ttftMs: null,
    promptTokens: null,
    completionTokens: null,
    predictedPerSecond: null,
  });
  const [attachments, setAttachments] = useState<Attachment[]>([]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  function saveSystemPrompt(value: string) {
    setSystemPrompt(value);
    if (selectedModel) {
      try {
        localStorage.setItem(`ananke-chat-sys-${selectedModel}`, value);
      } catch {
        // Best-effort.
      }
    }
  }

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  });

  async function send() {
    if (!selectedModel || !input.trim() || chatState.kind === "streaming")
      return;

    const userContent = input.trim();

    // Check if the model is running; start it if not.
    const svc = (services.data ?? []).find((s) => s.name === selectedModel);
    if (svc && svc.state !== "running") {
      setChatState({ kind: "starting" });
      try {
        await api.start(selectedModel);
        const deadline = Date.now() + 3 * 60 * 1000;
        while (Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 2000));
          const resp = await api.listServices();
          const s = resp.services.find((x) => x.name === selectedModel);
          if (s?.state === "running") break;
          if (s?.state === "failed") {
            setChatState({
              kind: "error",
              message: `service ${selectedModel} failed to start`,
            });
            return;
          }
        }
      } catch (e) {
        setChatState({
          kind: "error",
          message: `failed to start ${selectedModel}: ${e instanceof Error ? e.message : String(e)}`,
        });
        return;
      }
    }

    const imageAttachments = attachments.filter((a) => a.type === "image");
    const textAttachments = attachments.filter((a) => a.type === "text");

    let fullContent = userContent;
    for (const att of textAttachments) {
      fullContent += `\n\n${att.name}:\n${att.content}`;
    }

    const displayImages = imageAttachments.map((a) => a.content);

    setMessages([
      ...messages,
      {
        role: "user",
        content: userContent,
        images: displayImages.length > 0 ? displayImages : undefined,
      },
    ]);
    setAttachments([]);
    setStats({
      ttftMs: null,
      promptTokens: null,
      completionTokens: null,
      predictedPerSecond: null,
    });

    const controller = new AbortController();
    setChatState({ kind: "streaming", controller });

    const startTime = performance.now();
    let firstTokenTime: number | null = null;
    const finalStats: StreamStats = {
      ttftMs: null,
      promptTokens: null,
      completionTokens: null,
      predictedPerSecond: null,
    };

    function attachStats() {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, stats: { ...finalStats } };
        }
        return next;
      });
    }

    type ApiContentPart =
      | { type: "text"; text: string }
      | { type: "image_url"; image_url: { url: string } };

    type ApiMessage = {
      role: string;
      content: string | ApiContentPart[];
    };

    const apiMessages: ApiMessage[] = [];
    if (systemPrompt.trim()) {
      apiMessages.push({ role: "system", content: systemPrompt.trim() });
    }
    for (const m of messages) {
      apiMessages.push({ role: m.role, content: m.content });
    }
    if (imageAttachments.length > 0) {
      const parts: ApiContentPart[] = [{ type: "text", text: fullContent }];
      for (const img of imageAttachments) {
        parts.push({ type: "image_url", image_url: { url: img.content } });
      }
      apiMessages.push({ role: "user", content: parts });
    } else {
      apiMessages.push({ role: "user", content: fullContent });
    }

    try {
      const resp = await fetch(
        `${openaiBaseUrlFromListen(info.data?.openai_listen ?? "0.0.0.0:7070")}/v1/chat/completions`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          signal: controller.signal,
          body: JSON.stringify({
            model: selectedModel,
            messages: apiMessages,
            stream: true,
            stream_options: { include_usage: true },
          }),
        },
      );

      if (!resp.ok) {
        const text = await resp.text();
        setInput("");
        setChatState({
          kind: "error",
          message: `${resp.status} ${resp.statusText}: ${text}`,
        });
        return;
      }

      const reader = resp.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;
          const data = trimmed.slice(6);
          if (data === "[DONE]") continue;

          try {
            const chunk = JSON.parse(data) as {
              choices?: Array<{
                delta?: {
                  content?: string;
                  reasoning_content?: string;
                };
              }>;
              usage?: {
                prompt_tokens?: number;
                completion_tokens?: number;
              };
            };

            const delta = chunk.choices?.[0]?.delta;
            const contentDelta = delta?.content;
            const reasoningDelta = delta?.reasoning_content;

            if (contentDelta || reasoningDelta) {
              if (firstTokenTime === null) {
                firstTokenTime = performance.now();
                finalStats.ttftMs = firstTokenTime! - startTime;
                setStats({ ...finalStats });
                // Clear the textbox now that the model is responding.
                setInput("");
              }
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last?.role === "assistant") {
                  next[next.length - 1] = {
                    ...last,
                    content: contentDelta
                      ? last.content + contentDelta
                      : last.content,
                    reasoning: reasoningDelta
                      ? (last.reasoning ?? "") + reasoningDelta
                      : last.reasoning,
                  };
                } else {
                  next.push({
                    role: "assistant",
                    content: contentDelta ?? "",
                    reasoning: reasoningDelta || undefined,
                  });
                }
                return next;
              });
            }

            if (chunk.usage) {
              const elapsed = (performance.now() - startTime) / 1000;
              const completionTokens = chunk.usage.completion_tokens ?? 0;
              finalStats.promptTokens = chunk.usage?.prompt_tokens ?? null;
              finalStats.completionTokens = completionTokens;
              finalStats.predictedPerSecond =
                elapsed > 0 ? completionTokens / elapsed : null;
              setStats({ ...finalStats });
            }
          } catch {
            // Skip unparseable lines.
          }
        }
      }
      // Clear input in case no tokens arrived (empty response).
      setInput("");
      attachStats();
      setChatState({ kind: "idle" });
    } catch (e) {
      setInput("");
      attachStats();
      if (e instanceof DOMException && e.name === "AbortError") {
        setChatState({ kind: "idle" });
      } else {
        setChatState({
          kind: "error",
          message: e instanceof Error ? e.message : String(e),
        });
      }
    }
  }

  function cancel() {
    if (chatState.kind === "streaming") {
      chatState.controller.abort();
      setChatState({ kind: "idle" });
    }
  }

  function clearConversation() {
    setMessages([]);
    setStats({
      ttftMs: null,
      promptTokens: null,
      completionTokens: null,
      predictedPerSecond: null,
    });
    setChatState({ kind: "idle" });
  }

  async function handleFiles(files: FileList) {
    const selected = selectedModel
      ? (services.data ?? []).find((s) => s.name === selectedModel)
      : null;
    const hasVision = selected?.has_mmproj ?? false;

    for (const file of Array.from(files)) {
      if (file.type.startsWith("image/")) {
        if (!hasVision) continue;
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result;
          if (typeof result === "string") {
            setAttachments((prev) => [
              ...prev,
              {
                name: file.name,
                size: file.size,
                type: "image",
                content: result,
              },
            ]);
          }
        };
        reader.readAsDataURL(file);
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result;
          if (typeof result === "string") {
            setAttachments((prev) => [
              ...prev,
              {
                name: file.name,
                size: file.size,
                type: "text",
                content: result,
              },
            ]);
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

  const inputDisabled = !selectedModel || chatState.kind === "starting";

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex h-14 shrink-0 items-center border-b border-border-default px-4">
        <h1 className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-primary">
          Chat
        </h1>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-auto px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-tertiary">
            <span>Send a message to start chatting.</span>
            {chatState.kind === "starting" && (
              <span className="flex items-center gap-2">
                <Spinner />
                Starting {selectedModel}…
              </span>
            )}
          </div>
        ) : (
          messages.map((msg, i) => (
            <MessageBubble
              key={i}
              message={msg}
              modelName={selectedModel}
              liveStats={
                chatState.kind === "streaming" &&
                i === messages.length - 1 &&
                msg.role === "assistant"
                  ? stats
                  : null
              }
            />
          ))
        )}
        {messages.length > 0 && chatState.kind === "starting" && (
          <div className="flex items-center gap-2 py-2 text-sm text-tertiary">
            <Spinner />
            Starting {selectedModel}…
          </div>
        )}
        {chatState.kind === "error" && (
          <div className="mt-2 rounded-sm border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            {chatState.message}
          </div>
        )}
      </div>

      {/* System prompt */}
      {selectedModel && (
        <details className="border-t border-border-default px-4 py-2">
          <summary className="eyebrow cursor-pointer select-none hover:text-secondary">
            System prompt
          </summary>
          <textarea
            value={systemPrompt}
            onChange={(e) => saveSystemPrompt(e.target.value)}
            placeholder="You are a helpful assistant…"
            className="mt-1 h-20 w-full resize-none rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary placeholder:text-tertiary focus:border-accent focus:outline-none"
          />
        </details>
      )}

      {/* Attachments preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 border-t border-border-default px-4 py-2">
          {attachments.map((att, i) => (
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
                onClick={() =>
                  setAttachments((prev) => prev.filter((_, j) => j !== i))
                }
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
      <div className="border-t border-border-default px-4 py-3">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <ModelDropdown
            models={chatModels}
            selected={selectedModel}
            onSelect={selectModel}
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
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder={
              selectedModel
                ? chatState.kind === "starting"
                  ? "Starting model…"
                  : "Type a message… (Enter to send, Shift+Enter for newline)"
                : "Select a model first"
            }
            disabled={inputDisabled}
            rows={1}
            className="min-h-[40px] flex-1 resize-none rounded-sm border border-border-default bg-base px-2 py-2.5 text-sm text-primary placeholder:text-tertiary focus:border-accent focus:outline-none disabled:opacity-50"
          />
          <label className="flex w-10 shrink-0 cursor-pointer items-center justify-center rounded-md bg-elevated text-lg text-secondary hover:bg-border-strong">
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
          {chatState.kind === "streaming" || chatState.kind === "starting" ? (
            <button
              onClick={cancel}
              disabled={chatState.kind === "starting"}
              className="shrink-0 rounded-md bg-danger px-3 text-sm font-medium text-white hover:bg-danger/90 disabled:opacity-40"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={send}
              disabled={!selectedModel || !input.trim()}
              className="shrink-0 rounded-md bg-accent px-3 text-sm font-medium text-[var(--color-base)] hover:bg-accent/90 disabled:opacity-40"
            >
              Send
            </button>
          )}
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
          placeholder="Filter models…"
          className="h-7 w-full rounded-sm border border-border-default bg-surface px-2 text-sm text-primary placeholder:text-tertiary focus:border-accent focus:outline-none"
        />
      ) : (
        <button
          onClick={() => setOpen(true)}
          className="flex h-7 w-full items-center gap-2 rounded-sm border border-border-default bg-surface px-2 text-sm text-primary hover:bg-elevated"
        >
          {selectedSvc ? (
            <>
              <StatusDot state={selectedSvc.state} />
              <span className="font-mono">{selectedSvc.name}</span>
              {selectedSvc.has_mmproj && <Badge variant="vision">vision</Badge>}
            </>
          ) : (
            <span className="text-tertiary">select a model…</span>
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
              No matching models.
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
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isAssistant = message.role === "assistant";

  const label = isAssistant ? (modelName ?? "assistant") : message.role;
  const displayStats = liveStats ?? message.stats ?? null;

  return (
    <div className={`mb-4 ${isSystem ? "opacity-60" : ""}`}>
      <div className="mb-1 flex items-center gap-3">
        <span className="eyebrow">{label}</span>
        {isAssistant && displayStats && displayStats.promptTokens !== null && (
          <span className="flex items-center gap-3 text-xs text-tertiary">
            <span>prompt {displayStats.promptTokens} tokens</span>
            {displayStats.completionTokens !== null && (
              <span>output {displayStats.completionTokens} tokens</span>
            )}
            {displayStats.predictedPerSecond !== null && (
              <span>{displayStats.predictedPerSecond.toFixed(1)} tokens/s</span>
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
              reasoning
            </summary>
            <div className="mt-1 max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-xs text-secondary">
              {message.reasoning}
            </div>
          </details>
        )}
        {isAssistant ? (
          message.content ? (
            <div className="flex flex-col gap-3">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code: ({ children, className }) => {
                    const lang = className?.replace("language-", "");
                    return (
                      <pre className="overflow-x-auto bg-base p-2 font-mono text-xs">
                        <code data-lang={lang}>{children}</code>
                      </pre>
                    );
                  },
                }}
              >
                {DOMPurify.sanitize(message.content, {
                  USE_PROFILES: { html: true },
                })}
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

function ExternalLinkIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M15 3h6v6" />
      <path d="M10 14L21 3" />
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 6h18" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}
