// Logs viewer for a service. REST seed + WebSocket live tail, with
// stream filter, auto-follow, and a simple scroll container.
// Virtualisation will be added once the view is functional.

import { useEffect, useRef, useState } from "react";
import { useLogWindow, type LogWindow } from "../../api/hooks.ts";
import type { LogLine } from "../../api/client.ts";
import { formatTimestamp } from "../../util.ts";

type LogsViewerProps = {
  name: string;
};

const LIVE_WINDOW: LogWindow = { kind: "relative", durationMs: 5 * 60 * 1000 };

export function LogsViewer({ name }: LogsViewerProps) {
  const win = useLogWindow(name, LIVE_WINDOW);
  const [streamFilter, setStreamFilter] = useState<
    "both" | "stdout" | "stderr"
  >("both");
  const [autoFollow, setAutoFollow] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const lines = win.lines.filter(
    (l) => streamFilter === "both" || l.stream === streamFilter,
  );

  // Auto-follow: scroll to bottom when new lines arrive.
  useEffect(() => {
    if (autoFollow && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, autoFollow]);

  return (
    <div className="flex h-72 flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-2 border-b border-border-default px-3 py-1.5">
        <div className="flex items-center gap-0.5">
          {(["both", "stdout", "stderr"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStreamFilter(s)}
              className={`rounded-sm px-2 py-0.5 text-xs transition-colors ${
                streamFilter === s
                  ? "bg-elevated text-primary"
                  : "text-tertiary hover:text-secondary"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs text-tertiary">
          {win.isLive && <span className="text-success">live</span>}
          {win.droppedByOverflow > 0 && (
            <span className="text-warning">
              {win.droppedByOverflow} dropped
            </span>
          )}
          {win.truncated && <span className="text-warning">truncated</span>}
          <button
            onClick={() => setAutoFollow((v) => !v)}
            className={`rounded-sm px-2 py-0.5 transition-colors ${
              autoFollow ? "text-accent" : "text-tertiary hover:text-secondary"
            }`}
          >
            follow
          </button>
        </div>
      </div>

      {/* Log lines */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-auto px-3 py-1 font-mono text-xs"
      >
        {win.loading ? (
          <div className="text-tertiary">Loading…</div>
        ) : win.error ? (
          <div className="text-danger">{win.error.message}</div>
        ) : lines.length === 0 ? (
          <div className="flex min-h-[200px] items-center justify-center text-tertiary">
            No logs in this window.
          </div>
        ) : (
          lines.map((line) => (
            <LogRow key={`${line.run_id}:${line.seq}`} line={line} />
          ))
        )}
      </div>
    </div>
  );
}

function LogRow({ line }: { line: LogLine }) {
  return (
    <div className="flex gap-2 py-0.5 hover:bg-elevated">
      <span className="shrink-0 text-tertiary">
        {formatTimestamp(line.timestamp_ms)}
      </span>
      <span
        className={`shrink-0 ${
          line.stream === "stderr" ? "text-danger" : "text-tertiary"
        }`}
      >
        {line.stream === "stderr" ? "err" : "out"}
      </span>
      <span className="whitespace-pre-wrap break-all text-primary">
        {line.line}
      </span>
    </div>
  );
}
