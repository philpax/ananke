// One button that shows the theme it switches to. It is an action rather
// than an on/off state, so it has no `aria-pressed`.

import { useTranslation } from "react-i18next";

import { Button } from "../button/Button.tsx";
import { useTheme } from "./useTheme.ts";

export function ThemeToggle({ className }: { readonly className?: string }) {
  const { t } = useTranslation();
  const { theme, toggle } = useTheme();
  const label =
    theme === "dark" ? t("theme.switchTo.light") : t("theme.switchTo.dark");
  return (
    <Button
      tone="quiet"
      size="icon"
      icon={theme === "dark" ? "sun" : "moon"}
      aria-label={label}
      title={label}
      onClick={toggle}
      className={className}
    />
  );
}
