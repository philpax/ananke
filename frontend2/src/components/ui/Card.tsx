// Surface container with an optional engraved-label header. Used for
// cards, panels, and grouped content sections. The header is rendered
// as an `eyebrow` (tracked mono capitals) — the console's structural
// device — so every panel reads like a labelled instrument.

import type { ReactNode } from "react";

type CardProps = {
  header?: ReactNode;
  /** Optional element rendered at the right edge of the header row. */
  headerAction?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
};

export function Card({
  header,
  headerAction,
  children,
  className = "",
  bodyClassName = "",
}: CardProps) {
  return (
    <div
      className={`rounded-md border border-border-default bg-surface ${className}`}
    >
      {header !== undefined && (
        <div className="flex items-center gap-2 border-b border-border-default px-4 py-2.5">
          <span className="eyebrow">{header}</span>
          {headerAction && <div className="ml-auto">{headerAction}</div>}
        </div>
      )}
      <div className={bodyClassName || "p-4"}>{children}</div>
    </div>
  );
}
