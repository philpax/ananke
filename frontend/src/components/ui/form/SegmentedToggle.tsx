// A row of mutually exclusive options: time ranges, sort orders, stream
// filters. Each option is a button with `aria-pressed`, grouped under an
// accessible label.

import { cn } from "../cn.ts";

export type SegmentedOption<T> = {
  readonly label: string;
  readonly value: T;
};

type SegmentedToggleProps<T> = {
  readonly options: readonly SegmentedOption<T>[];
  readonly selected: T;
  readonly onChange: (value: T) => void;
  readonly label: string;
  readonly size?: "sm" | "md";
  readonly className?: string;
};

export function SegmentedToggle<T>({
  options,
  selected,
  onChange,
  label,
  size = "sm",
  className,
}: SegmentedToggleProps<T>) {
  return (
    <div
      role="group"
      aria-label={label}
      className={cn("flex flex-wrap items-center gap-0.5", className)}
    >
      {options.map((opt, index) => {
        const isSelected = Object.is(opt.value, selected);
        return (
          <button
            // Index, not label: two options may share a label.
            key={index}
            type="button"
            aria-pressed={isSelected}
            onClick={() => onChange(opt.value)}
            className={cn(
              "border px-2 text-xs font-medium",
              size === "sm" ? "h-control-sm" : "h-control",
              isSelected
                ? "border-accent bg-selected text-accent-text"
                : "border-transparent text-muted hover:bg-hover hover:text-ink",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
