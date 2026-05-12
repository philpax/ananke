import { useEffect, useRef, useState } from "react";

import { useLogWindow, type LogWindow } from "../api/hooks.ts";
import type { LogLine } from "../api/client.ts";

const PRESETS: ReadonlyArray<{ label: string; durationMs: number }> = [
  { label: "5m", durationMs: 5 * 60_000 },
  { label: "15m", durationMs: 15 * 60_000 },
  { label: "1h", durationMs: 60 * 60_000 },
  { label: "6h", durationMs: 6 * 60 * 60_000 },
  { label: "24h", durationMs: 24 * 60 * 60_000 },
];

const DEFAULT_PRESET = PRESETS[2]; // 1 h

/// Distance from the bottom (px) below which we consider the viewport
/// "pinned" and resume auto-following appended lines. Sized large
/// enough to absorb subpixel rounding from the browser's `scrollTop`
/// vs. `scrollHeight - clientHeight` reporting.
const PIN_THRESHOLD_PX = 24;

export function LogsView({ name }: { name: string }) {
  const [window, setWindow] = useState<LogWindow>({
    kind: "relative",
    durationMs: DEFAULT_PRESET.durationMs,
  });
  const result = useLogWindow(name, window);

  return (
    <div>
      <WindowPicker value={window} onChange={setWindow} />
      <StatusLine
        isLive={result.isLive}
        truncated={result.truncated}
        dropped={result.droppedByOverflow}
        error={result.error}
        count={result.lines.length}
      />
      <LogScroller lines={result.lines} loading={result.loading} />
    </div>
  );
}

function WindowPicker({
  value,
  onChange,
}: {
  value: LogWindow;
  onChange: (w: LogWindow) => void;
}) {
  const isAbsolute = value.kind === "absolute";
  const [expanded, setExpanded] = useState(isAbsolute);

  return (
    <div className="mb-2">
      <div className="flex flex-wrap gap-1 items-center text-xs">
        {PRESETS.map((p) => {
          const active =
            value.kind === "relative" && value.durationMs === p.durationMs;
          return (
            <button
              key={p.label}
              onClick={() => {
                setExpanded(false);
                onChange({ kind: "relative", durationMs: p.durationMs });
              }}
              className={`px-2 py-0.5 border rounded font-mono ${
                active
                  ? "border-blue-400 bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300"
                  : "border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`}
              title={`Last ${p.label}`}
            >
              Last {p.label}
            </button>
          );
        })}
        <button
          onClick={() => setExpanded((v) => !v)}
          className={`px-2 py-0.5 border rounded ${
            isAbsolute
              ? "border-blue-400 bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300"
              : "border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
          }`}
          title="Pick an explicit time range"
        >
          Custom range
        </button>
      </div>
      {expanded && (
        <AbsolutePicker
          value={value.kind === "absolute" ? value : null}
          onApply={(w) => onChange(w)}
        />
      )}
    </div>
  );
}

function AbsolutePicker({
  value,
  onApply,
}: {
  value: { sinceMs: number; untilMs: number | null } | null;
  onApply: (w: LogWindow) => void;
}) {
  // Lazy `useState` initialisers — they fire exactly once on mount,
  // sidestepping the react-hooks/purity rule that bans `Date.now()`
  // during render. The picker remounts when the user re-opens the
  // "Custom range" panel, which is the only time we want to re-anchor
  // to the current wall clock.
  const [from, setFrom] = useState(() =>
    toDatetimeLocal(value?.sinceMs ?? Date.now() - DEFAULT_PRESET.durationMs),
  );
  const [to, setTo] = useState(() =>
    value?.untilMs === null
      ? ""
      : toDatetimeLocal(value?.untilMs ?? Date.now()),
  );
  const [now, setNow] = useState(value?.untilMs === null);

  return (
    <div className="mt-2 p-2 border border-gray-200 dark:border-gray-800 rounded text-xs">
      <div className="grid grid-cols-[auto_1fr_auto_1fr_auto] gap-2 items-center">
        <label className="text-gray-500 dark:text-gray-400">From</label>
        <input
          type="datetime-local"
          step="1"
          value={from}
          onChange={(e) => setFrom(e.target.value)}
          className="border border-gray-300 dark:border-gray-700 rounded px-1 py-0.5 bg-white dark:bg-gray-900 dark:text-gray-100 font-mono"
        />
        <label className="text-gray-500 dark:text-gray-400">To</label>
        {now ? (
          <span className="font-mono text-gray-500 dark:text-gray-400">
            now (live)
          </span>
        ) : (
          <input
            type="datetime-local"
            step="1"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            className="border border-gray-300 dark:border-gray-700 rounded px-1 py-0.5 bg-white dark:bg-gray-900 dark:text-gray-100 font-mono"
          />
        )}
        <button
          onClick={() => {
            const since = fromDatetimeLocal(from);
            const until = now ? null : fromDatetimeLocal(to);
            if (since === null) return;
            if (!now && until === null) return;
            onApply({ kind: "absolute", sinceMs: since, untilMs: until });
          }}
          className="px-2 py-0.5 border border-blue-400 dark:border-blue-700 text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30 rounded hover:bg-blue-100 dark:hover:bg-blue-900/50"
        >
          Apply
        </button>
      </div>
      <label className="mt-2 inline-flex items-center gap-1 text-gray-600 dark:text-gray-300">
        <input
          type="checkbox"
          checked={now}
          onChange={(e) => setNow(e.target.checked)}
        />
        Stream live (no upper bound)
      </label>
    </div>
  );
}

function StatusLine({
  isLive,
  truncated,
  dropped,
  error,
  count,
}: {
  isLive: boolean;
  truncated: boolean;
  dropped: number;
  error: Error | null;
  count: number;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs mb-1">
      <span className="text-gray-500 dark:text-gray-400 tabular-nums">
        {count.toLocaleString()} lines
      </span>
      {isLive ? (
        <span className="inline-flex items-center gap-1 text-green-700 dark:text-green-400">
          <span className="inline-block w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
          Live
        </span>
      ) : (
        <span className="text-gray-500 dark:text-gray-400">Fixed range</span>
      )}
      {truncated && (
        <span
          className="text-amber-700 dark:text-amber-400"
          title="The window contains older lines than the seed page cap; refine the range to fetch more"
        >
          Older lines truncated
        </span>
      )}
      {dropped > 0 && (
        <span
          className="text-amber-700 dark:text-amber-400"
          title="The live stream dropped frames because the client lagged behind the broadcast buffer"
        >
          {dropped.toLocaleString()} live lines dropped
        </span>
      )}
      {error && (
        <span className="text-red-600 dark:text-red-400">
          Logs: {error.message}
        </span>
      )}
    </div>
  );
}

function LogScroller({
  lines,
  loading,
}: {
  lines: readonly LogLine[];
  loading: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  // `autoFollow` mirrors "scroll is parked at the bottom" — toggled by
  // the scroll handler, consumed by the post-render pin effect.
  const [autoFollow, setAutoFollow] = useState(true);

  // Track scroll position so we can flip auto-follow on/off based on
  // distance-from-bottom. Programmatic scrollTo writes from the pin
  // effect also fire this handler; they land at the bottom and keep
  // autoFollow == true, so there's no oscillation.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      setAutoFollow(distFromBottom <= PIN_THRESHOLD_PX);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // Pin to bottom whenever the line count changes and the user hasn't
  // scrolled away. Runs after layout so `scrollHeight` reflects the
  // freshly-appended line.
  useEffect(() => {
    if (!autoFollow) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines.length, autoFollow]);

  return (
    <div className="relative">
      <div
        ref={scrollRef}
        className="bg-gray-900 dark:bg-black text-gray-100 dark:text-gray-200 text-xs p-2 rounded h-96 overflow-y-auto whitespace-pre-wrap font-mono"
      >
        {loading && lines.length === 0 && (
          <div className="text-gray-500">Loading…</div>
        )}
        {!loading && lines.length === 0 && (
          <div className="text-gray-500">No logs in this window.</div>
        )}
        {lines.map((l) => (
          <LogRow key={`${l.run_id}:${l.seq}`} line={l} />
        ))}
      </div>
      {!autoFollow && (
        <button
          onClick={() => {
            const el = scrollRef.current;
            if (!el) return;
            el.scrollTop = el.scrollHeight;
            setAutoFollow(true);
          }}
          className="absolute right-3 bottom-3 px-2 py-1 text-xs border border-blue-400 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 rounded shadow hover:bg-blue-100 dark:hover:bg-blue-900/60"
          title="Resume auto-follow at the bottom"
        >
          Jump to latest ↓
        </button>
      )}
    </div>
  );
}

function LogRow({ line }: { line: LogLine }) {
  const ts = formatTimestamp(line.timestamp_ms);
  const color = line.stream === "stderr" ? "text-red-300" : "text-gray-100";
  return (
    <div className={color}>
      <span className="text-gray-500 dark:text-gray-600">{ts}</span> {line.line}
    </div>
  );
}

function formatTimestamp(ms: number): string {
  return new Date(ms).toISOString().slice(11, 23);
}

/// Convert a UNIX-ms timestamp to the `YYYY-MM-DDTHH:MM:SS` string the
/// browser's `<input type="datetime-local" step="1">` accepts. Local
/// timezone — the input is always interpreted as local.
function toDatetimeLocal(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

function fromDatetimeLocal(s: string): number | null {
  if (!s) return null;
  const d = new Date(s);
  const t = d.getTime();
  return Number.isFinite(t) ? t : null;
}
