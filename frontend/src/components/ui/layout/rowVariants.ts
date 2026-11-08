// Classes for a selectable row in a list: a severity rail on the left,
// replaced by the accent rail and a tinted fill when selected.

import { cn } from "../cn.ts";
import { toneRail, type Tone } from "../tone.ts";

export function rowVariants({
  selected = false,
  severity = "neutral",
}: {
  readonly selected?: boolean;
  readonly severity?: Tone;
} = {}): string {
  // `cn` lets the selected rail override the severity rail.
  return cn(
    toneRail({ tone: severity }),
    selected ? selectedClassName : unselectedClassName,
  );
}

const selectedClassName = "border-l-accent bg-selected-soft";
const unselectedClassName = "hover:bg-hover";
