// Copy-to-clipboard icon button. Its name says what it copies, the icon
// turns to a check on success, and the outcome (including a clipboard
// that is unavailable, as in a non-secure context) is announced. Clicks
// do not propagate, so it can sit inside a clickable row.

import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { cn } from "../cn.ts";
import { Icon } from "../icon/Icon.tsx";

type CopyStatus = "idle" | "copied" | "failed";

type CopyButtonProps = {
  readonly value: string;
  /// What is being copied, for the accessible name; defaults to `value`.
  readonly label?: string;
  readonly tabIndex?: number;
  readonly className?: string;
};

const RESET_AFTER_MS = 3_000;

export function CopyButton({
  value,
  label = value,
  tabIndex,
  className,
}: CopyButtonProps) {
  const { t } = useTranslation();
  const [status, setStatus] = useState<CopyStatus>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setStatus("idle"), RESET_AFTER_MS);
  }

  const name =
    status === "copied"
      ? t("common.copiedValue", { label })
      : status === "failed"
        ? t("common.copyFailed", { label })
        : t("common.copyValue", { label });
  const announcement =
    status === "copied"
      ? t("common.copied")
      : status === "failed"
        ? t("common.clipboardUnavailable")
        : "";

  return (
    <button
      type="button"
      title={name}
      aria-label={name}
      tabIndex={tabIndex}
      onKeyDown={(e) => e.stopPropagation()}
      onClick={(e) => {
        e.stopPropagation();
        void copy();
      }}
      className={cn(
        "inline-flex size-6 shrink-0 items-center justify-center hover:text-ink",
        status === "copied" ? "text-ok" : "text-soft",
        className,
      )}
    >
      <Icon name={status === "copied" ? "check" : "copy"} size="3.5" />
      <span role="status" className="sr-only">
        {announcement}
      </span>
    </button>
  );
}
