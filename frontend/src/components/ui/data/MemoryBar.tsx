// A memory budget as a filled track: a device's resident and pledged
// memory, or one service's share of a device beside everything else on
// it. The bar is an image to assistive technology; `label` describes it.

import { cn } from "../cn.ts";
import {
  memorySegments,
  type MemoryBarInput,
  type MemorySegmentKind,
} from "./memorySegments.ts";

type MemoryBarProps = MemoryBarInput & {
  readonly label: string;
  readonly size?: "sm" | "md";
  /// Placement only: the service needs an eviction to fit, so its share
  /// is drawn as a warning.
  readonly needsEviction?: boolean;
  readonly className?: string;
};

export function MemoryBar({
  label,
  size = "sm",
  needsEviction = false,
  className,
  ...input
}: MemoryBarProps) {
  const segments = memorySegments(input);
  return (
    <div
      role="img"
      aria-label={label}
      title={label}
      className={cn(
        "flex border border-line-strong bg-canvas",
        size === "sm" ? "h-2" : "h-3",
        className,
      )}
    >
      {segments.map((segment) => (
        <div
          key={segment.kind}
          className={cn(
            SEGMENT_CLASSES[segment.kind],
            needsEviction && WARN_CLASSES[segment.kind],
          )}
          style={{ width: `${segment.percent}%` }}
        />
      ))}
    </div>
  );
}

const SEGMENT_CLASSES: Readonly<Record<MemorySegmentKind, string>> = {
  used: "bg-accent-data",
  pledged: "bg-line-strong",
  others: "bg-line-strong",
  self: "bg-accent-data",
  growth: "bg-accent-data/40",
};

const WARN_CLASSES: Readonly<Record<MemorySegmentKind, string>> = {
  used: "",
  pledged: "",
  others: "",
  self: "bg-warn",
  growth: "bg-warn/40",
};
