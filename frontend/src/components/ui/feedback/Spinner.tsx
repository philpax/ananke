// A small accent-coloured spinner with a screen-reader label.

import { useTranslation } from "react-i18next";

import { cn } from "../cn.ts";

type SpinnerProps = {
  readonly size?: "3.5" | "4";
  /// Read by screen readers; defaults to "Loading…".
  readonly label?: string;
  readonly className?: string;
};

export function Spinner({ size = "4", label, className }: SpinnerProps) {
  const { t } = useTranslation();
  return (
    <span
      role="status"
      className={cn("inline-flex shrink-0 text-accent", className)}
    >
      <svg
        aria-hidden="true"
        className={cn("animate-spin", size === "4" ? "size-4" : "size-3.5")}
        viewBox="0 0 24 24"
        fill="none"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="3"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      <span className="sr-only">{label ?? t("common.loading")}</span>
    </span>
  );
}
