// Logs viewer for a service. REST seed + WebSocket live tail, with
// time window presets, custom time range, text search, run filter,
// stream filter, auto-follow, and a scroll container.
//
// Aims for GCP Log Explorer parity on the essentials: jump-to-time,
// text search within the buffer, run filtering, and live tail.

import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLogWindow, type LogWindow } from "../../api/hooks.ts";
import type { LogLine } from "../../api/client.ts";
import { formatTimestamp } from "../../util.ts";
import { SegmentedToggle } from "../ui/SegmentedToggle.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { EmptyState } from "../ui/EmptyState.tsx";

type LogsViewerProps = {
  name: string;
};

type WindowMode = "preset" | "custom";

type WindowPreset = {
  label: string;
  window: LogWindow;
};

const WINDOW_PRESETS: WindowPreset[] = [
  { label: "5m", window: { kind: "relative", durationMs: 5 * 60 * 1000 } },
  { label: "1h", window: { kind: "relative", durationMs: 60 * 60 * 1000 } },
  { label: "6h", window: { kind: "relative", durationMs: 6 * 60 * 60 * 1000 } },
  {
    label: "24h",
    window: { kind: "relative", durationMs: 24 * 60 * 60 * 1000 },
  },
];

// Convert a datetime-local string (e.g. "2026-06-24T14:30") to ms
// since epoch. Returns null for empty/invalid input.
function datetimeToMs(dt: string): number | null {
  if (!dt) return null;
  const t = new Date(dt).getTime();
  return Number.isNaN(t) ? null : t;
}

// Convert ms since epoch to a datetime-local string suitable for an
// <input type="datetime-local"> value.
function msToDatetime(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function LogsViewer({ name }: LogsViewerProps) {
  const { t } = useTranslation();
  const [windowMode, setWindowMode] = useState<WindowMode>("preset");
  const [presetIdx, setPresetIdx] = useState(0);

  // Custom time range state. Empty until the user selects "custom".
  const [sinceInput, setSinceInput] = useState("");
  const [untilInput, setUntilInput] = useState("");

  const window: LogWindow = useMemo(() => {
    if (windowMode === "preset") return WINDOW_PRESETS[presetIdx]!.window;
    const since = datetimeToMs(sinceInput);
    const until = untilInput ? datetimeToMs(untilInput) : null;
    if (since === null) return { kind: "relative", durationMs: 5 * 60 * 1000 };
    return { kind: "absolute", sinceMs: since, untilMs: until };
  }, [windowMode, presetIdx, sinceInput, untilInput]);

  const win = useLogWindow(name, window);
  const [streamFilter, setStreamFilter] = useState<
    "both" | "stdout" | "stderr"
  >("both");
  const [search, setSearch] = useState("");
  const [runFilter, setRunFilter] = useState<number | null>(null);
  const [autoFollow, setAutoFollow] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Collect unique run IDs from loaded lines.
  const runIds = useMemo(() => {
    const set = new Set<number>();
    for (const l of win.lines) set.add(l.run_id);
    return [...set].sort((a, b) => b - a);
  }, [win.lines]);

  const lines = useMemo(() => {
    let result = win.lines;
    if (streamFilter !== "both")
      result = result.filter((l) => l.stream === streamFilter);
    if (runFilter !== null)
      result = result.filter((l) => l.run_id === runFilter);
    if (search) {
      const lower = search.toLowerCase();
      result = result.filter((l) => l.line.toLowerCase().includes(lower));
    }
    return result;
  }, [win.lines, streamFilter, runFilter, search]);

  // Auto-follow: scroll to bottom when new lines arrive.
  useEffect(() => {
    if (autoFollow && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, autoFollow]);

  // When switching to custom mode, pre-fill the since field if empty.
  function selectCustom() {
    if (windowMode !== "custom") {
      const now = Date.now();
      setSinceInput(msToDatetime(now - 60 * 60 * 1000));
      setUntilInput("");
    }
    setWindowMode("custom");
  }

  return (
    <div className="flex h-72 flex-col">
      {/* Toolbar row 1: time + stream + status */}
      <div className="flex items-center gap-2 border-b border-border-default px-4 py-2">
        <SegmentedToggle<number | "custom">
          options={[
            ...WINDOW_PRESETS.map((preset, i) => ({
              label: preset.label,
              value: i as number,
            })),
            { label: t("logs.custom"), value: "custom" as const },
          ]}
          selected={windowMode === "preset" ? presetIdx : "custom"}
          onChange={(v) => {
            if (v === "custom") {
              selectCustom();
            } else {
              setPresetIdx(v);
              setWindowMode("preset");
            }
          }}
        />
        <div className="mx-1 h-3 w-px bg-border-default" />
        <SegmentedToggle<"both" | "stdout" | "stderr">
          options={(["both", "stdout", "stderr"] as const).map((s) => ({
            label: t(`logs.${s}`),
            value: s,
          }))}
          selected={streamFilter}
          onChange={setStreamFilter}
        />
        <div className="ml-auto flex items-center gap-2 text-xs text-tertiary">
          {win.isLive && <span className="text-success">{t("logs.live")}</span>}
          {win.droppedByOverflow > 0 && (
            <span className="text-warning">
              {t("logs.dropped", { value: win.droppedByOverflow })}
            </span>
          )}
          {win.truncated && (
            <span className="text-warning">{t("logs.truncated")}</span>
          )}
          <button
            onClick={() => setAutoFollow((v) => !v)}
            className={`rounded-sm px-2 py-0.5 transition-colors ${
              autoFollow ? "text-accent" : "text-tertiary hover:text-secondary"
            }`}
          >
            {t("logs.follow")}
          </button>
        </div>
      </div>

      {/* Toolbar row 2: custom time range (shown only in custom mode) */}
      {windowMode === "custom" && (
        <div className="flex items-center gap-2 border-b border-border-default px-4 py-2">
          <input
            type="datetime-local"
            value={sinceInput}
            onChange={(e) => setSinceInput(e.target.value)}
            className="h-6 rounded-sm border border-border-default bg-base px-1.5 text-xs text-primary focus:border-accent focus:outline-none"
          />
          <span className="text-xs text-tertiary">→</span>
          <input
            type="datetime-local"
            value={untilInput}
            onChange={(e) => setUntilInput(e.target.value)}
            placeholder={t("logs.now")}
            className="h-6 rounded-sm border border-border-default bg-base px-1.5 text-xs text-primary focus:border-accent focus:outline-none"
          />
          {!untilInput && (
            <span className="text-xs text-tertiary">{t("logs.liveParen")}</span>
          )}
        </div>
      )}

      {/* Toolbar row 3: search + run filter */}
      <div className="flex items-center gap-2 border-b border-border-default px-4 py-2">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={t("logs.search")}
          className="h-6 w-40 rounded-sm border border-border-default bg-base px-2 text-xs text-primary placeholder:text-tertiary focus:border-accent focus:outline-none"
        />
        {runIds.length > 1 && (
          <select
            value={runFilter ?? ""}
            onChange={(e) =>
              setRunFilter(e.target.value ? Number(e.target.value) : null)
            }
            className="h-6 rounded-sm border border-border-default bg-base px-1.5 text-xs text-primary focus:border-accent focus:outline-none"
          >
            <option value="">{t("logs.allRuns")}</option>
            {runIds.map((r) => (
              <option key={r} value={r}>
                {t("logs.run", { value: r })}
              </option>
            ))}
          </select>
        )}
        <span className="ml-auto font-mono text-xs text-tertiary">
          {t("logs.lines", { value: lines.length })}
        </span>
      </div>

      {/* Log lines */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-auto px-4 py-2 font-mono text-xs"
      >
        {win.loading ? (
          <Spinner />
        ) : win.error ? (
          <div className="text-danger">{win.error.message}</div>
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

function LogRow({ line, search }: { line: LogLine; search: string }) {
  const { t } = useTranslation();
  return (
    <div className="flex gap-2 py-0.5 hover:bg-elevated">
      <span className="shrink-0 text-tertiary">
        {formatTimestamp(line.timestamp_ms)}
      </span>
      <span className="shrink-0 text-tertiary">r{line.run_id}</span>
      <span
        className={`shrink-0 ${
          line.stream === "stderr" ? "text-danger" : "text-tertiary"
        }`}
      >
        {line.stream === "stderr" ? t("logs.err") : t("logs.out")}
      </span>
      <span className="whitespace-pre-wrap break-all text-primary">
        {search ? highlightSearch(line.line, search) : line.line}
      </span>
    </div>
  );
}

// Split text around search matches and wrap matches in a <mark>.
function highlightSearch(text: string, search: string): React.ReactNode {
  if (!search) return text;
  const lower = text.toLowerCase();
  const query = search.toLowerCase();
  const parts: React.ReactNode[] = [];
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
      <mark key={key++} className="bg-warning/30 text-primary">
        {text.slice(idx, idx + query.length)}
      </mark>,
    );
    i = idx + query.length;
  }
  return parts;
}
