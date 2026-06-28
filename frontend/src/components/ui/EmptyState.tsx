// Empty state: icon + message for empty lists and missing data.

import type { ReactNode } from "react";

type EmptyStateProps = {
  icon?: ReactNode;
  message: ReactNode;
  className?: string;
};

export function EmptyState({ icon, message, className = "" }: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-2 py-8 text-center ${className}`}
    >
      {icon && <span className="text-tertiary">{icon}</span>}
      <p className="text-sm text-tertiary">{message}</p>
    </div>
  );
}
