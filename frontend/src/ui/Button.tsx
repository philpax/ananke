import clsx from "clsx";
import { twMerge } from "tailwind-merge";
import type { JSX } from "preact";

import { Icon } from "./Icon";
import type { IconName } from "./Icon";

export function Button({
  children,
  icon,
  tone = "neutral",
  class: className,
  ...props
}: JSX.IntrinsicElements["button"] & {
  readonly icon?: IconName;
  readonly tone?: "neutral" | "primary" | "start" | "stop";
}) {
  return (
    <button
      type="button"
      {...props}
      class={twMerge(
        clsx(
          `
            inline-flex h-8 shrink-0 items-center justify-center gap-[7px]
            border px-3 text-[13px] font-medium
            hover:brightness-115
            disabled:opacity-40
          `,
          {
            "border-line-strong bg-raised text-ink": tone === "neutral",
            "border-accent-strong bg-accent-strong text-(--text-on-accent)":
              tone === "primary",
            "border-ok-line bg-ok-bg text-ok": tone === "start",
            "border-danger-line bg-danger-bg text-danger": tone === "stop",
          },
          className,
        ),
      )}
    >
      {icon && <Icon name={icon} size={15} />} {children}
    </button>
  );
}
