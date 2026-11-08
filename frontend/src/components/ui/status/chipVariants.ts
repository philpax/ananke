import { cva, type VariantProps } from "class-variance-authority";

import { toneBox } from "../tone.ts";

/// Status tones take the filled `toneBox` look; `accent` and the
/// modalities are outlined on a transparent fill so they read as tags
/// rather than status.
export const chipVariants = cva(
  "inline-flex items-center border px-1 text-xs whitespace-nowrap",
  {
    variants: {
      tone: {
        neutral: toneBox({ tone: "neutral" }),
        ok: toneBox({ tone: "ok" }),
        warn: toneBox({ tone: "warn" }),
        danger: toneBox({ tone: "danger" }),
        accent: "border-line-hover text-accent",
        vision: "border-vision-line text-vision",
        embedding: "border-embedding-line text-embedding",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export type ChipTone = NonNullable<VariantProps<typeof chipVariants>["tone"]>;
