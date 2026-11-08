// A bordered surface, optionally headed by a `SectionHeading` row.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";
import { SectionHeading } from "./SectionHeading.tsx";

type PanelProps = {
  readonly as?: "section" | "div";
  readonly heading?: ReactNode;
  readonly actions?: ReactNode;
  readonly padding?: "none" | "md";
  readonly className?: string;
  readonly children: ReactNode;
};

export function Panel({
  as: Tag = "section",
  heading,
  actions,
  padding = "md",
  className,
  children,
}: PanelProps) {
  return (
    <Tag className={cn("border border-line bg-surface", className)}>
      {heading !== undefined && (
        <SectionHeading
          actions={actions}
          className="flex min-h-control-lg items-center border-b border-line px-4 py-1.5"
        >
          {heading}
        </SectionHeading>
      )}
      {padding === "md" ? <div className="p-4">{children}</div> : children}
    </Tag>
  );
}
