// A short tag: a status word, a modality, or a flag such as "elastic".

import type { ReactNode } from "react";

import { cn } from "../cn.ts";
import { chipVariants, type ChipTone } from "./chipVariants.ts";

export function Chip({
  tone = "neutral",
  className,
  children,
}: {
  readonly tone?: ChipTone;
  readonly className?: string;
  readonly children: ReactNode;
}) {
  return (
    <span className={cn(chipVariants({ tone }), className)}>{children}</span>
  );
}
