// One message in the transcript on its role's surface: a header with who
// wrote it, when, the reply's token figures, and a copy button, then the
// reasoning, the text, and any images.

import { useTranslation } from "react-i18next";

import { formatTokenRate } from "../../../util.ts";
import { cn } from "../../ui/cn.ts";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { replyRates } from "../chatMeta.ts";
import { Markdown } from "../markdown/Markdown.tsx";
import type { Message, StreamStats } from "../state/chatStore.ts";
import { Reasoning } from "./Reasoning.tsx";

type TurnProps = {
  readonly message: Message;
  /// The stats of a reply still streaming, which its message gains at the
  /// end.
  readonly liveStats: StreamStats | null;
};

export function Turn({ message, liveStats }: TurnProps) {
  const { t } = useTranslation();
  const user = message.role !== "assistant";
  const stats = liveStats ?? message.stats ?? null;
  return (
    <article
      className={cn(
        "border",
        user ? "border-user-line bg-user" : "border-model-line bg-model",
      )}
    >
      <header
        className={cn(
          "flex flex-wrap items-center gap-x-2.5 gap-y-0.5 border-b py-1 pr-1 pl-3",
          user
            ? "border-user-line-soft bg-hover"
            : "border-model-line bg-model-raised",
        )}
      >
        <h2
          className={cn(
            "min-w-0 text-sm wrap-anywhere text-ink",
            user
              ? "font-semibold tracking-wider uppercase"
              : "font-mono font-bold",
          )}
        >
          {user ? t("chat.you") : (message.model ?? t("chat.assistant"))}
        </h2>
        <time
          className="font-mono text-xs text-muted"
          dateTime={message.timestamp}
          title={message.timestamp}
        >
          {message.timestamp.slice(11, 19)}
        </time>
        {!user && stats && <ReplyStats stats={stats} />}
        <CopyButton
          value={message.content}
          label={user ? t("chat.copyMessage") : t("chat.copyReply")}
          className="ml-auto"
        />
      </header>
      <div className="p-3">
        {message.reasoning && <Reasoning text={message.reasoning} />}
        {user ? (
          <p className="text-md wrap-break-word whitespace-pre-wrap text-ink">
            {message.content}
          </p>
        ) : (
          message.content && <Markdown text={message.content} />
        )}
        {message.images && message.images.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {message.images.map((src, i) => (
              <img
                key={i}
                src={src}
                alt={t("chat.imageAlt", { index: i + 1 })}
                className="max-h-40 max-w-full border border-line object-cover"
              />
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

function ReplyStats({ stats }: { readonly stats: StreamStats }) {
  const { t } = useTranslation();
  if (stats.promptTokens === null) return null;
  const rates = replyRates(stats);
  const parts = [
    t("chat.tokensIn", { value: stats.promptTokens }),
    stats.completionTokens !== null &&
      t("chat.tokensOut", { value: stats.completionTokens }),
    rates.kind === "split" &&
      t("chat.inputTokensPerSecond", { value: formatTokenRate(rates.input) }),
    rates.kind === "split" &&
      t("chat.outputTokensPerSecond", {
        value: formatTokenRate(rates.output),
      }),
    rates.kind === "effective" &&
      t("chat.tokensPerSecond", { value: formatTokenRate(rates.rate) }),
  ].filter((part) => typeof part === "string");
  return (
    <span className="font-mono text-xs text-body">{parts.join(" · ")}</span>
  );
}
