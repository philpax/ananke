import type { ComponentProps } from "react";

import { cn } from "../cn.ts";
import { fieldVariants, type FieldSize } from "./fieldVariants.ts";

type TextInputProps = Readonly<
  Omit<ComponentProps<"input">, "size"> & {
    size?: Exclude<FieldSize, "auto">;
    mono?: boolean;
  }
>;

export function TextInput({
  size = "md",
  mono = false,
  type = "text",
  className,
  ...props
}: TextInputProps) {
  return (
    <input
      type={type}
      className={cn(fieldVariants({ size, mono }), className)}
      {...props}
    />
  );
}
