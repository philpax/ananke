// Classes for one navigation item: a labelled row with an icon in the
// rail, a touch-sized tab in the narrow tab row. The active item takes
// the selected fill and an accent border.

import { cva } from "class-variance-authority";

export const navItemVariants = cva(
  "flex shrink-0 items-center gap-2.5 border px-3 text-sm max-md:h-touch md:h-control-lg md:px-2.5",
  {
    variants: {
      active: {
        true: "border-accent bg-selected font-semibold text-accent-text",
        false: "border-transparent text-body hover:bg-hover hover:text-ink",
      },
      /// Pushed to the bottom of the rail.
      pinned: {
        true: "md:mt-auto",
        false: "",
      },
    },
    defaultVariants: { active: false, pinned: false },
  },
);
