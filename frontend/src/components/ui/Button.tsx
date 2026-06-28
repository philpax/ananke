// Button primitive with variants and sizes. Icon + label composition
// via children.

import type { ButtonHTMLAttributes, ReactNode } from "react";

import {
  buttonClassName,
  type ButtonSize,
  type ButtonVariant,
} from "./buttonStyles.ts";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
  children: ReactNode;
};

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  children,
  ...props
}: ButtonProps) {
  return (
    <button className={buttonClassName(variant, size, className)} {...props}>
      {children}
    </button>
  );
}
