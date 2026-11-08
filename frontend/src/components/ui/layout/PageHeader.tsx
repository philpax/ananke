// The bar that caps every view: a title, optional meta beside it, and
// actions pushed to the right. Wraps rather than truncating on narrow
// screens.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";

type PageHeaderProps = {
  readonly title: ReactNode;
  readonly meta?: ReactNode;
  readonly actions?: ReactNode;
  readonly className?: string;
};

export function PageHeader({
  title,
  meta,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <header
      className={cn(
        "flex min-h-header shrink-0 flex-wrap items-center gap-3 border-b border-line bg-surface px-4 py-2.5",
        className,
      )}
    >
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1.5">
        <h1 className="text-lg font-semibold text-ink">{title}</h1>
        {meta && (
          <span className="text-sm wrap-anywhere text-muted">{meta}</span>
        )}
      </div>
      {actions && (
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {actions}
        </div>
      )}
    </header>
  );
}
