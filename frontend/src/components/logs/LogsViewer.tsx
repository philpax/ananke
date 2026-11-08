// A service's logs over the shared time range: the newest page over HTTP,
// then, for a rolling range, a follow on the live socket. A fixed range
// shows that window without following. Scrolling to the top fetches older
// lines a page at a time.

import { useState } from "react";
import { useTranslation } from "react-i18next";

import { useLogWindow } from "../../api/logs/useLogWindow.ts";
import { useStickToBottom } from "../../hooks/useStickToBottom.ts";
import { EmptyState } from "../ui/feedback/EmptyState.tsx";
import { Message } from "../ui/feedback/Message.tsx";
import { logWindowFor, type TimeRange } from "../ui/form/timeRange.ts";
import { LogRow } from "./LogRow.tsx";
import { LogToolbar } from "./LogToolbar.tsx";
import { filterLines, type StreamFilter } from "./logFilter.ts";

export function LogsViewer({
  name,
  range,
}: {
  readonly name: string;
  readonly range: TimeRange;
}) {
  const { t } = useTranslation();
  const win = useLogWindow(name, logWindowFor(range));
  const [stream, setStream] = useState<StreamFilter>("both");
  const [search, setSearch] = useState("");
  const [run, setRun] = useState<number | null>(null);

  const runIds = [...new Set(win.lines.map((l) => l.run_id))].sort(
    (a, b) => b - a,
  );
  const lines = filterLines(win.lines, stream, run, search);

  // Stick to the bottom as lines arrive while pinned there; scrolling up
  // detaches, and scrolling back down or Follow re-attaches.
  const { scrollRef, onScroll, pinned, scrollToBottom } =
    useStickToBottom(lines);
  const onListScroll = () => {
    onScroll();
    if (scrollRef.current?.scrollTop === 0 && win.hasOlder) win.loadOlder();
  };

  return (
    <div className="flex h-96 flex-col">
      <LogToolbar
        status={{
          fixed: range.kind === "fixed",
          live: win.isLive,
          skipped: win.skipped,
          truncated: win.truncated,
          lineCount: lines.length,
        }}
        stream={stream}
        onStream={setStream}
        search={search}
        onSearch={setSearch}
        runIds={runIds}
        run={run}
        onRun={setRun}
        pinned={pinned}
        onFollow={scrollToBottom}
      />
      <div
        ref={scrollRef}
        onScroll={onListScroll}
        className="min-h-0 flex-1 overflow-auto px-4 py-2 font-mono text-xs"
      >
        {win.hasOlder && !win.loading && (
          <button
            type="button"
            onClick={win.loadOlder}
            disabled={win.loadingOlder}
            className="mb-1 text-muted hover:text-ink disabled:opacity-40"
          >
            {win.loadingOlder ? t("logs.loadingOlder") : t("logs.loadOlder")}
          </button>
        )}
        {win.loading ? (
          <Message kind="loading" />
        ) : win.error ? (
          <Message kind="error" text={win.error.message} />
        ) : lines.length === 0 ? (
          <EmptyState message={t("logs.emptyState")} />
        ) : (
          lines.map((line) => (
            <LogRow
              key={`${line.run_id}:${line.seq}`}
              line={line}
              search={search}
            />
          ))
        )}
      </div>
    </div>
  );
}
