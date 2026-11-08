// The time range the service history, its logs, and the Events and Stats
// views share, kept in the URL: `?range=1h` for a rolling preset that ends
// now, or `?from=<ms>&to=<ms>` for a fixed range. No parameter means the
// default preset. The URL is the only store, so a link or a reload keeps
// the range, and the nav carries it between those views.

import type { LogWindow } from "../../../api/logs/logWindow.ts";

export const RANGE_PRESETS = [
  { id: "5m", durationMs: 5 * 60 * 1000 },
  { id: "1h", durationMs: 60 * 60 * 1000 },
  { id: "6h", durationMs: 6 * 60 * 60 * 1000 },
  { id: "24h", durationMs: 24 * 60 * 60 * 1000 },
] as const;

export type RangePreset = (typeof RANGE_PRESETS)[number]["id"];

export type TimeRange =
  | { readonly kind: "rolling"; readonly preset: RangePreset }
  | { readonly kind: "fixed"; readonly fromMs: number; readonly toMs: number };

export const DEFAULT_RANGE = {
  kind: "rolling",
  preset: "5m",
} as const satisfies TimeRange;

/// The search parameters a range owns; everything else in the query is
/// left alone.
export const RANGE_PARAMS = ["range", "from", "to"] as const;

/// The range a query names. Anything malformed, including a fixed range
/// whose start is not before its end, falls back to the default rather
/// than showing an empty or inverted window.
export function parseTimeRange(params: URLSearchParams): TimeRange {
  const from = params.get("from");
  const to = params.get("to");
  if (from !== null || to !== null) {
    const fromMs = parseMs(from);
    const toMs = parseMs(to);
    if (fromMs === null || toMs === null || fromMs >= toMs) {
      return DEFAULT_RANGE;
    }
    return { kind: "fixed", fromMs, toMs };
  }
  const preset = RANGE_PRESETS.find((p) => p.id === params.get("range"));
  return preset === undefined ? DEFAULT_RANGE : rolling(preset.id);
}

/// `params` with its range parameters replaced by `range`'s. The default
/// range writes none, so the plain URL stays plain.
export function withTimeRange(
  params: URLSearchParams,
  range: TimeRange,
): URLSearchParams {
  const next = new URLSearchParams(params);
  for (const key of RANGE_PARAMS) next.delete(key);
  if (range.kind === "fixed") {
    next.set("from", String(range.fromMs));
    next.set("to", String(range.toMs));
  } else if (range.preset !== DEFAULT_RANGE.preset) {
    next.set("range", range.preset);
  }
  return next;
}

/// The range part of a location's search, as a `?…` suffix for a link, or
/// `""` when it holds none. Used to carry the range across navigation.
export function rangeSearch(search: string): string {
  const range = withTimeRange(
    new URLSearchParams(),
    parseTimeRange(new URLSearchParams(search)),
  ).toString();
  return range === "" ? "" : `?${range}`;
}

/// The range's bounds at `nowMs`. A rolling range has no end: it runs to
/// now and keeps following.
export function resolveTimeRange(
  range: TimeRange,
  nowMs: number,
): { readonly sinceMs: number; readonly untilMs: number | null } {
  if (range.kind === "fixed") {
    return { sinceMs: range.fromMs, untilMs: range.toMs };
  }
  return { sinceMs: nowMs - presetDuration(range.preset), untilMs: null };
}

/// The logs window for a range: a rolling range follows new lines, a fixed
/// one does not.
export function logWindowFor(range: TimeRange): LogWindow {
  return range.kind === "fixed"
    ? { kind: "absolute", sinceMs: range.fromMs, untilMs: range.toMs }
    : { kind: "relative", durationMs: presetDuration(range.preset) };
}

/// A stable identity for a range, for keys and memo dependencies.
export function rangeKey(range: TimeRange): string {
  return range.kind === "fixed"
    ? `${range.fromMs}-${range.toMs}`
    : range.preset;
}

/// A short label for a range: the preset's name, or `custom`.
export function rangeLabel(
  range: TimeRange,
  t: (key: string) => string,
): string {
  return range.kind === "fixed" ? t("common.custom") : range.preset;
}

/// A `datetime-local` input's value as ms since the epoch, or `null` when
/// it is empty or invalid.
export function datetimeToMs(value: string): number | null {
  if (!value) return null;
  const ms = new Date(value).getTime();
  return Number.isNaN(ms) ? null : ms;
}

/// Ms since the epoch as a `datetime-local` input's value, in local time.
export function msToDatetime(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function presetDuration(preset: RangePreset): number {
  return (
    RANGE_PRESETS.find((p) => p.id === preset)?.durationMs ??
    RANGE_PRESETS[0].durationMs
  );
}

function rolling(preset: RangePreset): TimeRange {
  return { kind: "rolling", preset };
}

function parseMs(raw: string | null): number | null {
  if (raw === null || !/^\d+$/.test(raw)) return null;
  const ms = Number(raw);
  return Number.isSafeInteger(ms) ? ms : null;
}
