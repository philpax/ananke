// A link that looks like a `Button`: `to` for an in-app route, or `href`
// for a plain anchor. An external anchor opens in a new tab without a
// referrer and gains the `open` icon after its label.

import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { cn } from "../cn.ts";
import { Icon } from "../icon/Icon.tsx";
import type { IconName } from "../icon/iconGeometry.tsx";
import {
  buttonIconSize,
  buttonVariants,
  type ButtonSize,
  type ButtonTone,
} from "./buttonVariants.ts";

type Target =
  | { readonly to: string; readonly href?: never; readonly external?: never }
  | { readonly href: string; readonly external?: boolean; readonly to?: never };

type ButtonLinkProps = Target & {
  readonly tone?: ButtonTone;
  readonly size?: ButtonSize;
  readonly icon?: IconName;
  readonly title?: string;
  readonly "aria-label"?: string;
  readonly className?: string;
  readonly children?: ReactNode;
};

export function ButtonLink({
  tone = "neutral",
  size = "md",
  icon,
  className,
  children,
  ...rest
}: ButtonLinkProps) {
  const classes = cn(buttonVariants({ tone, size }), className);
  const iconSize = buttonIconSize(size);
  const leading = icon && <Icon name={icon} size={iconSize} />;
  const label = { title: rest.title, "aria-label": rest["aria-label"] };

  if (rest.to !== undefined) {
    return (
      <Link to={rest.to} className={classes} {...label}>
        {leading}
        {children}
      </Link>
    );
  }
  return (
    <a
      href={rest.href}
      className={classes}
      {...label}
      {...(rest.external ? { target: "_blank", rel: "noreferrer" } : {})}
    >
      {leading}
      {children}
      {rest.external && <Icon name="open" size={iconSize} />}
    </a>
  );
}
