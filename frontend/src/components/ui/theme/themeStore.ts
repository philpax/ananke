// The imperative shell around `theme.ts`: a module-level store that
// reads the stored override and the OS preference, writes `data-theme`
// on <html>, and keeps both in sync across tabs. Read it through
// `useTheme`.
//
// An OS change leaves an existing override alone, even when the OS now
// matches it; only the user's toggle clears the stored key.

import {
  THEME_STORAGE_KEY,
  nextOverride,
  parseOverride,
  type ThemeOverride,
} from "./theme.ts";

export type ThemeState = {
  readonly override: ThemeOverride;
  readonly systemDark: boolean;
};

export function subscribeTheme(listener: () => void): () => void {
  ensureStarted();
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/// Returns the same object until the state changes, as
/// `useSyncExternalStore` requires.
export function getThemeState(): ThemeState {
  return ensureStarted();
}

export function toggleTheme(): void {
  const current = ensureStarted();
  const override = nextOverride(current.override, current.systemDark);
  writeStorage(override);
  setState({ ...current, override });
}

const DARK_QUERY = "(prefers-color-scheme: dark)";

const listeners = new Set<() => void>();
let state: ThemeState | null = null;

function ensureStarted(): ThemeState {
  if (state) return state;
  const media = darkQuery();
  state = { override: readStorage(), systemDark: media?.matches ?? false };
  media?.addEventListener("change", (event) => {
    if (state) setState({ ...state, systemDark: event.matches });
  });
  window.addEventListener("storage", (event) => {
    if (state && event.key === THEME_STORAGE_KEY) {
      setState({ ...state, override: parseOverride(event.newValue) });
    }
  });
  applyToDocument(state.override);
  return state;
}

function setState(next: ThemeState): void {
  state = next;
  // The attribute is written before listeners run, so anything that reads
  // computed colours on notification (the chart) sees the new theme.
  applyToDocument(next.override);
  for (const listener of listeners) listener();
}

function applyToDocument(override: ThemeOverride): void {
  const root = document.documentElement;
  if (override) {
    root.dataset.theme = override;
    root.style.colorScheme = override;
  } else {
    delete root.dataset.theme;
    root.style.removeProperty("color-scheme");
  }
}

function darkQuery(): MediaQueryList | null {
  return typeof window.matchMedia === "function"
    ? window.matchMedia(DARK_QUERY)
    : null;
}

// Storage can throw (private windows, blocked site data); the theme then
// lasts only for the page's lifetime.
function readStorage(): ThemeOverride {
  try {
    return parseOverride(localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return null;
  }
}

function writeStorage(override: ThemeOverride): void {
  try {
    if (override) localStorage.setItem(THEME_STORAGE_KEY, override);
    else localStorage.removeItem(THEME_STORAGE_KEY);
  } catch {
    // See `readStorage`.
  }
}
