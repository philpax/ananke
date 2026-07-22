// ViewHeader: the fixed-height rule that caps every view. Renders the
// shared flex header with a bottom border; children supply the title,
// toolbar, and stats. The canonical gap is `gap-3` — kept as the default
// `className` rather than baked into the base so views that need a
// wider or tighter rhythm (e.g. `gap-5`, `gap-2`) can override it
// without colliding with a second gap utility.

import type { ReactNode } from "react";

type ViewHeaderProps = {
  children: ReactNode;
  className?: string;
};

export function ViewHeader({ children, className = "gap-3" }: ViewHeaderProps) {
  return (
    <header
      className={`flex min-h-[var(--header-height)] shrink-0 flex-wrap items-center border-b border-border-default px-4 py-3 pt-[calc(env(safe-area-inset-top)+0.75rem)] ${className}`}
    >
      {children}
    </header>
  );
}
