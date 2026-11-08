// A section or panel title in sentence case, with optional actions to its
// right.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";

type SectionHeadingProps = {
  readonly level?: 2 | 3;
  readonly actions?: ReactNode;
  readonly className?: string;
  readonly children: ReactNode;
};

export function SectionHeading({
  level = 2,
  actions,
  className,
  children,
}: SectionHeadingProps) {
  const Tag = level === 2 ? "h2" : "h3";
  const heading = (
    <Tag className="min-w-0 text-sm font-semibold text-ink">{children}</Tag>
  );
  if (!actions) {
    return className ? <div className={className}>{heading}</div> : heading;
  }
  return (
    <div className={cn("flex items-center gap-2", className)}>
      {heading}
      <div className="ml-auto flex items-center gap-2">{actions}</div>
    </div>
  );
}
