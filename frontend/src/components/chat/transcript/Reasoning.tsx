// A reply's reasoning trace: a disclosure whose capped scroll box follows
// the trace as it streams, the way the transcript does, until the reader
// scrolls up inside it.

import { useTranslation } from "react-i18next";

import { useStickToBottom } from "../../../hooks/useStickToBottom.ts";
import { Markdown } from "../markdown/Markdown.tsx";

export function Reasoning({ text }: { readonly text: string }) {
  const { t } = useTranslation();
  const { scrollRef, onScroll } = useStickToBottom(text);
  const lines = text.trimEnd().split("\n").length;
  return (
    <details open className="mb-3 border border-line bg-raised">
      <summary className="cursor-pointer px-2.5 py-1.5 text-sm text-body select-none hover:text-ink">
        {t("chat.reasoningLines", { count: lines })}
      </summary>
      <div
        ref={scrollRef}
        onScroll={onScroll}
        role="region"
        tabIndex={0}
        aria-label={t("chat.reasoning")}
        className="max-h-60 overflow-y-auto border-t border-line px-2.5 py-2"
      >
        <Markdown text={text} className="text-sm text-body" />
      </div>
    </details>
  );
}
