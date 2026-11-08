// Shared classes for text inputs, selects, and textareas. The border uses
// `line-control`, the one line token at 3:1, because a field's boundary
// is how it is recognised (WCAG 1.4.11).

import { cva, type VariantProps } from "class-variance-authority";

export const fieldVariants = cva(
  "min-w-0 border border-line-control bg-canvas px-2.5 text-sm text-ink placeholder:text-faint focus-visible:border-accent disabled:opacity-40",
  {
    variants: {
      size: {
        sm: "h-control-sm",
        md: "h-control",
        lg: "h-control-lg",
        // Textareas grow with their content rather than taking a height.
        auto: "py-2",
      },
      mono: { true: "font-mono", false: "" },
    },
    defaultVariants: { size: "md", mono: false },
  },
);

export type FieldSize = NonNullable<VariantProps<typeof fieldVariants>["size"]>;
