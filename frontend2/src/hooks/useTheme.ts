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

function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  const prefersLight = window.matchMedia(
    "(prefers-color-scheme: light)",
  ).matches;
  const isLight = theme === "light" || (theme === "system" && prefersLight);
  root.classList.toggle("light", isLight);
}

export function useTheme(): {
  theme: Theme;
  setTheme: (t: Theme) => void;
  isLight: boolean;
} {
  const [theme, setThemeState] = useState<Theme>(getStoredTheme);

  useEffect(() => {
    applyTheme(theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // Best-effort persistence.
    }
  }, [theme]);

  // Listen for system theme changes when in "system" mode.
  useEffect(() => {
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const handler = () => applyTheme("system");
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [theme]);

  const setTheme = useCallback((t: Theme) => setThemeState(t), []);

  const prefersLight = window.matchMedia(
    "(prefers-color-scheme: light)",
  ).matches;
  const isLight = theme === "light" || (theme === "system" && prefersLight);

  return { theme, setTheme, isLight };
}
