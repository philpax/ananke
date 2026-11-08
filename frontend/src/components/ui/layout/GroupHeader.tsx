// A sticky band that heads a group of rows, such as services sharing a
// state. One of the two places tracked uppercase is used; the other is
// `ColumnHeader`.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";
import { StatusDot } from "../status/StatusDot.tsx";
import type { Tone } from "../tone.ts";

type GroupHeaderProps = {
  readonly tone?: Tone;
  readonly count?: number;
  readonly sticky?: boolean;
  readonly className?: string;
  readonly children: ReactNode;
};

export function GroupHeader({
  tone = "neutral",
  count,
  sticky = true,
  className,
  children,
}: GroupHeaderProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 border-y border-line bg-raised px-4 py-1 text-2xs font-semibold tracking-wider text-muted uppercase",
        sticky && "sticky top-0 z-10",
        className,
      )}
    >
      <StatusDot tone={tone} className="size-1.5" />
      <span className="min-w-0 truncate">{children}</span>
      {count !== undefined && (
        <span className="font-mono text-faint">{count}</span>
      )}
    </div>
  );
}
