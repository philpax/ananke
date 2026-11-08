import { useSyncExternalStore } from "react";

import { resolveTheme, type Theme, type ThemeOverride } from "./theme.ts";
import { getThemeState, subscribeTheme, toggleTheme } from "./themeStore.ts";

export function useTheme(): {
  readonly theme: Theme;
  readonly override: ThemeOverride;
  readonly toggle: () => void;
} {
  const state = useSyncExternalStore(subscribeTheme, getThemeState);
  return {
    theme: resolveTheme(state.override, state.systemDark),
    override: state.override,
    toggle: toggleTheme,
  };
}
