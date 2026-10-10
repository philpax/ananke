import type { Range } from "../api/client";

export function presetRange(hours: number, now = Date.now()): Range {
  return { since: now - hours * 3_600_000, until: now, preset: hours };
}

export function localDate(ms: number) {
  const date = new Date(ms);
  return new Date(ms - date.getTimezoneOffset() * 60_000)
    .toISOString()
    .slice(0, 16);
}

export function parseRange(from: string, to: string): Range | string {
  const since = new Date(from).getTime(),
    until = new Date(to).getTime();
  if (!Number.isFinite(since) || !Number.isFinite(until))
    return "Enter a valid start and end date.";
  if (until <= since) return "End must be after start.";
  return { since, until, preset: null };
}
