// The label row above a grid of data rows. The caller passes the same
// `grid-cols-*` its rows use, so labels line up with their columns.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";

export function ColumnHeader({
  className,
  children,
}: {
  readonly className?: string;
  readonly children: ReactNode;
}) {
  return (
    <div
      className={cn(
        "grid items-center gap-3 border-b border-line bg-surface px-4 py-1.5 text-2xs font-semibold tracking-wider text-muted uppercase",
        className,
      )}
    >
      {children}
    </div>
  );
}
