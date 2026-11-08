// A fenced code block: its language and a copy button above the
// highlighted code.

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import type { CodeBlockInfo } from "./codeBlock.ts";

export function CodeBlock({
  info,
  children,
}: {
  readonly info: CodeBlockInfo;
  readonly children: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <div className="mb-3 border border-line bg-canvas last:mb-0">
      <div className="flex items-center justify-between gap-2 border-b border-line bg-raised py-0.5 pr-1 pl-2">
        <span className="font-mono text-xs text-muted">
          {info.language ?? t("chat.plainText")}
        </span>
        <CopyButton value={info.text} label={t("chat.copyCode")} />
      </div>
      <pre className="overflow-x-auto p-3 font-mono text-xs/relaxed text-body">
        {children}
      </pre>
    </div>
  );
}
