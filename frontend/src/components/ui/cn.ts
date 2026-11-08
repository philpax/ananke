// Class-name composition: `clsx` for conditionals, `tailwind-merge` so a
// caller's `className` overrides a primitive's default instead of both
// landing in the output.

import { clsx, type ClassValue } from "clsx";
import { extendTailwindMerge } from "tailwind-merge";

export function cn(...inputs: readonly ClassValue[]): string {
  return twMerge(clsx(inputs));
}

// The named spacing entries in `styles/theme.css` must be listed here, or
// tailwind-merge cannot tell that `h-control` and `h-8` conflict and
// keeps both.
const twMerge = extendTailwindMerge({
  extend: {
    theme: {
      text: ["2xs", "md"],
      spacing: [
        "control-sm",
        "control",
        "control-lg",
        "touch",
        "header",
        "band",
        "band-identity",
        "nav",
      ],
    },
  },
});
