// A native checkbox with its label. The native control follows the
// theme through `color-scheme`; `accent-color` tints it.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";

type CheckboxProps = {
  readonly label: ReactNode;
  readonly checked: boolean;
  readonly onChange: (checked: boolean) => void;
  readonly disabled?: boolean;
  readonly className?: string;
};

export function Checkbox({
  label,
  checked,
  onChange,
  disabled,
  className,
}: CheckboxProps) {
  return (
    <label
      className={cn(
        "inline-flex items-center gap-2 text-sm text-body has-disabled:opacity-40",
        className,
      )}
    >
      <input
        type="checkbox"
        className="size-4 accent-accent-strong"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}
