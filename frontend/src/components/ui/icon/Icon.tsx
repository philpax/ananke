// Decorative icon. Always `aria-hidden`: the control it sits in carries
// the accessible name.

import { cn } from "../cn.ts";
import { ICON_GEOMETRY, type IconName } from "./iconGeometry.tsx";

export type IconSize = "3" | "3.5" | "4" | "5";

type IconProps = {
  readonly name: IconName;
  readonly size?: IconSize;
  readonly filled?: boolean;
  readonly className?: string;
};

export function Icon({
  name,
  size = "4",
  filled = false,
  className,
}: IconProps) {
  const geometry = ICON_GEOMETRY[name];
  return (
    <svg
      aria-hidden="true"
      className={cn("shrink-0", SIZE_CLASSES[size], className)}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={geometry.strokeWidth}
      strokeLinejoin={geometry.strokeLinejoin}
      strokeLinecap={geometry.strokeLinecap}
    >
      {geometry.shape}
    </svg>
  );
}

const SIZE_CLASSES: Readonly<Record<IconSize, string>> = {
  "3": "size-3",
  "3.5": "size-3.5",
  "4": "size-4",
  "5": "size-5",
};
