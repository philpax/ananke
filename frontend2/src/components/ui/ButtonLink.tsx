// Link styled as a Button. Shares the same variant/size classes so
// links and buttons can sit in the same toolbar without visual
// mismatch.

import type { ReactNode } from "react";
import { Link } from "react-router-dom";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

type ButtonLinkProps = {
  to: string;
  variant?: Variant;
  size?: Size;
  className?: string;
  children: ReactNode;
};

const VARIANT_CLASSES: Record<Variant, string> = {
  primary: "bg-accent text-[var(--color-base)] hover:bg-accent/90",
  secondary: "bg-elevated text-primary hover:bg-border-strong",
  ghost: "text-secondary hover:text-primary hover:bg-elevated",
  danger: "bg-danger text-white hover:bg-danger/90",
};

const SIZE_CLASSES: Record<Size, string> = {
  sm: "h-7 px-2 text-xs",
  md: "h-8 px-3 text-sm",
};

export function ButtonLink({
  to,
  variant = "secondary",
  size = "md",
  className = "",
  children,
}: ButtonLinkProps) {
  return (
    <Link
      to={to}
      className={`inline-flex items-center gap-1.5 rounded-md font-medium transition-colors ${VARIANT_CLASSES[variant]} ${SIZE_CLASSES[size]} ${className}`}
    >
      {children}
    </Link>
  );
}
