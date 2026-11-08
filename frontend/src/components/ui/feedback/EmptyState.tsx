// A centred icon and message for an empty list or a missing dataset.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";
import { Icon } from "../icon/Icon.tsx";
import type { IconName } from "../icon/iconGeometry.tsx";

type EmptyStateProps = {
  readonly message: ReactNode;
  readonly icon?: IconName;
  readonly className?: string;
};

export function EmptyState({
  message,
  icon = "empty",
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 py-8 text-center text-muted",
        className,
      )}
    >
      <Icon name={icon} size="5" className="text-faint" />
      <p className="text-sm">{message}</p>
    </div>
  );
}
