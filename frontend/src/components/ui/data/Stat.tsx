// A labelled figure: a muted label over a mono value, optionally with a
// unit. `xl` is the summary-stat size.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";

type StatProps = {
  readonly label: ReactNode;
  readonly value: ReactNode;
  readonly unit?: string;
  readonly size?: "sm" | "xl";
  readonly className?: string;
};

export function Stat({
  label,
  value,
  unit,
  size = "sm",
  className,
}: StatProps) {
  return (
    <div className={className}>
      <div className="text-xs text-muted">{label}</div>
      <div
        className={cn(
          "mt-0.5 font-mono text-ink",
          size === "xl" ? "text-xl" : "text-sm",
        )}
      >
        {value}
        {unit && <span className="ml-0.5 text-muted">{unit}</span>}
      </div>
    </div>
  );
}
