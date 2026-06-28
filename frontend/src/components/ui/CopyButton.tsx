// Copy-to-clipboard button. Shows a clipboard icon by default, a check
// icon on success. Inline next to values, with text-tertiary colour so
// it doesn't compete with the data itself.

import { useState, useCallback } from "react";
import { useTranslation } from "react-i18next";

type CopyButtonProps = {
  value: string;
  className?: string;
};

export function CopyButton({ value, className = "" }: CopyButtonProps) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const onClick = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1_500);
    } catch {
      // Clipboard API may be unavailable (non-secure context).
    }
  }, [value]);

  return (
    <button
      type="button"
      onClick={onClick}
      title={copied ? t("common.copied") : t("common.copy")}
      className={`inline-flex items-center text-tertiary hover:text-secondary transition-colors ${className}`}
    >
      {copied ? (
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
          <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
        </svg>
      )}
    </button>
  );
}
