import type { ComponentProps } from "react";

import { cn } from "../cn.ts";
import { fieldVariants } from "./fieldVariants.ts";

type TextareaProps = Readonly<ComponentProps<"textarea"> & { mono?: boolean }>;

export function Textarea({ mono = false, className, ...props }: TextareaProps) {
  return (
    <textarea
      className={cn(fieldVariants({ size: "auto", mono }), className)}
      {...props}
    />
  );
}
