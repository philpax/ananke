// One log line: time, run, stream marker, and the text with search
// matches highlighted. Only stderr reads as a problem; a container's
// combined output carries ordinary lines too.

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import type { LogLine } from "../../api/client.ts";
import { formatTimestamp } from "../../util.ts";
import { cn } from "../ui/cn.ts";

export function LogRow({
  line,
  search,
}: {
  readonly line: LogLine;
  readonly search: string;
}) {
  const { t } = useTranslation();
  const stream =
    line.stream === "stderr"
      ? t("logs.err")
      : line.stream === "combined"
        ? t("logs.combined")
        : t("logs.out");
  return (
    <div className="flex gap-2 py-0.5 hover:bg-hover">
      <span className="shrink-0 text-faint max-sm:hidden">
        {formatTimestamp(line.timestamp_ms)}
      </span>
      <span className="shrink-0 text-faint">r{line.run_id}</span>
      <span
        className={cn(
          "shrink-0",
          line.stream === "stderr" ? "text-danger" : "text-faint",
        )}
      >
        {stream}
      </span>
      <span className="min-w-0 break-all whitespace-pre-wrap text-ink">
        {search ? highlightSearch(line.line, search) : line.line}
      </span>
    </div>
  );
}

function highlightSearch(text: string, search: string): ReactNode[] {
  const lower = text.toLowerCase();
  const query = search.toLowerCase();
  const parts: ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < text.length) {
    const idx = lower.indexOf(query, i);
    if (idx === -1) {
      parts.push(text.slice(i));
      break;
    }
    if (idx > i) parts.push(text.slice(i, idx));
    parts.push(
      <mark key={key++} className="bg-warn-bg text-ink">
        {text.slice(idx, idx + query.length)}
      </mark>,
    );
    i = idx + query.length;
  }
  return parts;
}
