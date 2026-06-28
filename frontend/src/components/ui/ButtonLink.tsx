// Link styled as a Button. Shares the same variant/size classes so
// links and buttons can sit in the same toolbar without visual
// mismatch.

import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import {
  buttonClassName,
  type ButtonSize,
  type ButtonVariant,
} from "./buttonStyles.ts";

type ButtonLinkProps = {
  to: string;
  variant?: ButtonVariant;
  size?: ButtonSize;
  className?: string;
  children: ReactNode;
};

export function ButtonLink({
  to,
  variant = "secondary",
  size = "md",
  className = "",
  children,
}: ButtonLinkProps) {
  return (
    <Link to={to} className={buttonClassName(variant, size, className)}>
      {children}
    </Link>
  );
}
