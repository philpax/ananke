// The resizable service list's width rules. The stored width is clamped
// here; the cap at half the view is CSS-only, so nothing measures it.

export const LIST_WIDTH_MIN = 240;
export const LIST_WIDTH_MAX = 600;
export const LIST_WIDTH_DEFAULT = 324;

export type ListPrefs = {
  readonly width: number;
  readonly hidden: boolean;
};

export function clampListWidth(width: number): number {
  if (!Number.isFinite(width)) return LIST_WIDTH_DEFAULT;
  return Math.round(Math.min(LIST_WIDTH_MAX, Math.max(LIST_WIDTH_MIN, width)));
}

/// The width a separator key moves to: arrows step 16 px (64 with Shift),
/// Home and End go to the bounds. `null` for a key that does nothing.
export function stepListWidth(
  width: number,
  key: string,
  shift: boolean,
): number | null {
  const step = shift ? 64 : 16;
  switch (key) {
    case "ArrowLeft":
      return clampListWidth(width - step);
    case "ArrowRight":
      return clampListWidth(width + step);
    case "Home":
      return LIST_WIDTH_MIN;
    case "End":
      return LIST_WIDTH_MAX;
    default:
      return null;
  }
}

/// Stored preferences, read defensively: anything that is not what was
/// written falls back to the default for that field.
export function parseListPrefs(
  width: string | null,
  hidden: string | null,
): ListPrefs {
  const parsed = width === null ? Number.NaN : Number(width);
  return {
    width: Number.isFinite(parsed)
      ? clampListWidth(parsed)
      : LIST_WIDTH_DEFAULT,
    hidden: hidden === "true",
  };
}
