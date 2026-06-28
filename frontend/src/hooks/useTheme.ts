// Dark/light/system theme management. Dark is the default; the choice
// is persisted in localStorage. When set to "system", follows
// prefers-color-scheme via a media query listener.

import { useCallback, useEffect, useState } from "react";

type Theme = "dark" | "light" | "system";
const STORAGE_KEY = "ananke-theme";

function getStoredTheme(): Theme {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "dark" || v === "light" || v === "system") return v;
  } catch {
    // localStorage may be unavailable (private mode, etc.).
  }
  return "dark";
}

function applyTheme(theme: Theme, prefersLight: boolean): void {
  const root = document.documentElement;
  const isLight = theme === "light" || (theme === "system" && prefersLight);
  root.classList.toggle("light", isLight);
}

function readPrefersLight(): boolean {
  return window.matchMedia("(prefers-color-scheme: light)").matches;
}

export function useTheme(): {
  theme: Theme;
  setTheme: (t: Theme) => void;
  isLight: boolean;
} {
  const [theme, setThemeState] = useState<Theme>(getStoredTheme);
  const [prefersLight, setPrefersLight] = useState<boolean>(() =>
    readPrefersLight(),
  );

  useEffect(() => {
    applyTheme(theme, prefersLight);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // Best-effort persistence.
    }
  }, [theme, prefersLight]);

  // Listen for system theme changes so `isLight` stays current in
  // "system" mode. The handler updates state, not the effect body.
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const handler = (e: MediaQueryListEvent) => setPrefersLight(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const setTheme = useCallback((t: Theme) => setThemeState(t), []);

  const isLight = theme === "light" || (theme === "system" && prefersLight);

  return { theme, setTheme, isLight };
}
