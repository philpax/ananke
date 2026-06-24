// Surface container with optional header. Used for cards, panels,
// and grouped content sections.

import type { ReactNode } from "react";

type CardProps = {
  header?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
};

export function Card({
  header,
  children,
  className = "",
  bodyClassName = "",
}: CardProps) {
  return (
    <div
      className={`rounded-md border border-border-default bg-surface ${className}`}
    >
      {header !== undefined && (
        <div className="border-b border-border-default px-3 py-2 text-sm font-medium text-primary">
          {header}
        </div>
      )}
      <div className={`p-3 ${bodyClassName}`}>{children}</div>
    </div>
  );
}
