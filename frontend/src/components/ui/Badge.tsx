// Coloured pill badge for states, modalities, and fit verdicts.
// Semantic colours map to the theme tokens.

import type { ReactNode } from "react";

export type BadgeVariant =
  | "neutral"
  | "success"
  | "warning"
  | "danger"
  | "accent"
  | "vision"
  | "embedding";

type BadgeProps = {
  variant?: BadgeVariant;
  children: ReactNode;
  className?: string;
};

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  neutral: "bg-elevated text-secondary ring-border-strong/60",
  success: "bg-success/12 text-success ring-success/25",
  warning: "bg-warning/12 text-warning ring-warning/25",
  danger: "bg-danger/12 text-danger ring-danger/25",
  accent: "bg-accent/12 text-accent ring-accent/25",
  vision: "bg-vision/12 text-vision ring-vision/25",
  embedding: "bg-embedding/12 text-embedding ring-embedding/25",
};

// Modality/state tags use the eyebrow register — tracked mono capitals —
// so they read as engraved labels rather than rounded web pills.
export function Badge({
  variant = "neutral",
  children,
  className = "",
}: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-[3px] px-1.5 py-0.5 font-mono text-[0.625rem] font-medium uppercase tracking-[0.08em] ring-1 ring-inset ${VARIANT_CLASSES[variant]} ${className}`}
    >
      {children}
    </span>
  );
}
