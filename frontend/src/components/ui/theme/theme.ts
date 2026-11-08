// Theme rules, after https://lea.verou.me/blog/2026/dark-mode-toggles/:
// with no stored choice the page follows the OS; only an override is
// stored; toggling to the theme the OS already asks for clears it, so the
// page goes back to following the OS.
//
// The boot script in `index.html` repeats the storage key and the parse;
// `boot.test.ts` keeps the two in step.

export type Theme = "light" | "dark";
export type ThemeOverride = Theme | null;

export const THEME_STORAGE_KEY = "ananke-theme";

export function parseOverride(raw: string | null): ThemeOverride {
  return raw === "light" || raw === "dark" ? raw : null;
}

export function resolveTheme(
  override: ThemeOverride,
  systemDark: boolean,
): Theme {
  return override ?? systemTheme(systemDark);
}

/// The override after one press of the toggle: the opposite of what is
/// showing, or none when that opposite is what the OS asks for anyway.
export function nextOverride(
  override: ThemeOverride,
  systemDark: boolean,
): ThemeOverride {
  const next: Theme =
    resolveTheme(override, systemDark) === "dark" ? "light" : "dark";
  return next === systemTheme(systemDark) ? null : next;
}

function systemTheme(systemDark: boolean): Theme {
  return systemDark ? "dark" : "light";
}
