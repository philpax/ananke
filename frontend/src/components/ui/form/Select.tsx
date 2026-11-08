import type { ComponentProps } from "react";

import { cn } from "../cn.ts";
import { fieldVariants, type FieldSize } from "./fieldVariants.ts";

type SelectProps = Readonly<
  Omit<ComponentProps<"select">, "size"> & {
    size?: Exclude<FieldSize, "auto">;
  }
>;

export function Select({ size = "md", className, ...props }: SelectProps) {
  return (
    <select
      className={cn(fieldVariants({ size }), "pr-1.5", className)}
      {...props}
    />
  );
}
