// Shared classes for button-like elements (`Button`, `ButtonLink`), so a
// link and a button in one toolbar look the same. Tones are named by
// meaning: `primary` is the one main action in a group, `start` and
// `stop` are lifecycle actions, `quiet` recedes into a toolbar, and
// everything else is `neutral`.

import { cva, type VariantProps } from "class-variance-authority";

import type { IconSize } from "../icon/Icon.tsx";
import { toneBox } from "../tone.ts";

export const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-1.5 border font-medium whitespace-nowrap hover:brightness-115 disabled:opacity-40 disabled:hover:brightness-100",
  {
    variants: {
      tone: {
        neutral: "border-line-strong bg-raised text-ink",
        primary: "border-accent-strong bg-accent-strong text-on-accent",
        start: toneBox({ tone: "ok" }),
        stop: toneBox({ tone: "danger" }),
        quiet:
          "border-transparent bg-transparent text-soft hover:bg-hover hover:text-ink hover:brightness-100",
      },
      size: {
        sm: "h-control-sm px-2 text-xs",
        md: "h-control px-3 text-sm",
        "icon-sm": "size-control-sm",
        icon: "size-control",
      },
    },
    defaultVariants: { tone: "neutral", size: "md" },
  },
);

export type ButtonTone = NonNullable<
  VariantProps<typeof buttonVariants>["tone"]
>;
export type ButtonSize = NonNullable<
  VariantProps<typeof buttonVariants>["size"]
>;

export function buttonIconSize(size: ButtonSize): IconSize {
  return size === "sm" || size === "icon-sm" ? "3.5" : "4";
}
