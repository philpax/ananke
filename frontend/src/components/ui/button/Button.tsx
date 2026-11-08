// Button with a semantic tone, a size, and an optional leading icon.
// `type` defaults to "button" so a button inside a form never submits it
// by accident.

import type { ComponentProps } from "react";

import { cn } from "../cn.ts";
import { Icon } from "../icon/Icon.tsx";
import type { IconName } from "../icon/iconGeometry.tsx";
import {
  buttonIconSize,
  buttonVariants,
  type ButtonSize,
  type ButtonTone,
} from "./buttonVariants.ts";

type ButtonProps = Readonly<
  ComponentProps<"button"> & {
    tone?: ButtonTone;
    size?: ButtonSize;
    icon?: IconName;
  }
>;

export function Button({
  tone = "neutral",
  size = "md",
  icon,
  type = "button",
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(buttonVariants({ tone, size }), className)}
      {...props}
    >
      {icon && <Icon name={icon} size={buttonIconSize(size)} />}
      {children}
    </button>
  );
}
