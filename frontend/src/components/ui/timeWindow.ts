// Shared time-window model + helpers for the TimeWindowSelect
// component and its consumers (logs, stats, dashboard). Kept separate
// from the component file so react-refresh sees a components-only
// module there.

import type { LogWindow } from "../../api/hooks.ts";

/// The selector's value: identical to the logs API's window shape so
/// consumers can pass it straight through.
export type TimeWindow = LogWindow;

export type TimeWindowPreset = { label: string; durationMs: number };

export const TIME_WINDOW_PRESETS: TimeWindowPreset[] = [
  { label: "5m", durationMs: 5 * 60 * 1000 },
  { label: "1h", durationMs: 60 * 60 * 1000 },
  { label: "6h", durationMs: 6 * 60 * 60 * 1000 },
  { label: "24h", durationMs: 24 * 60 * 60 * 1000 },
];

// Human label for a window: the matching preset label for relative
// windows ("5m", "1h", ...), or a duration-ish fallback, or "custom"
// for absolute ranges.
export function windowLabel(w: TimeWindow, t: (key: string) => string): string {
  if (w.kind === "absolute") return t("common.custom");
  const preset = TIME_WINDOW_PRESETS.find((p) => p.durationMs === w.durationMs);
  if (preset) return preset.label;
  const mins = Math.round(w.durationMs / 60_000);
  return mins < 60 ? `${mins}m` : `${Math.round(mins / 60)}h`;
}

// Convert a datetime-local input value to ms since epoch. Returns null
// for empty/invalid input.
export function datetimeToMs(dt: string): number | null {
  if (!dt) return null;
  const t = new Date(dt).getTime();
  return Number.isNaN(t) ? null : t;
}

// Convert ms since epoch to a datetime-local string suitable for an
// <input type="datetime-local"> value.
export function msToDatetime(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
