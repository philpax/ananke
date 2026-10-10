import clsx from "clsx";
import { twMerge } from "tailwind-merge";

import { Icon } from "../ui/Icon";
import type { Section } from "./navigation";

const sections: readonly Section[] = [
  "services",
  "chat",
  "events",
  "stats",
  "config",
];

export function Nav({
  current,
  onNavigate,
  compact = false,
}: {
  readonly current: Section;
  readonly onNavigate: (section: Section) => void;
  readonly compact?: boolean;
}) {
  return (
    <nav
      aria-label="Sections"
      class={twMerge(
        clsx(
          `flex shrink-0 gap-0.5 bg-surface`,
          compact
            ? `overflow-x-auto border-b border-line px-1.5 py-[5px]`
            : `w-44 flex-col border-r-2 border-line-strong px-2 py-2.5`,
        ),
      )}
    >
      {sections.map((section) => (
        <button
          key={section}
          type="button"
          aria-current={current === section ? "page" : undefined}
          class={twMerge(
            clsx(
              `
                flex h-[38px] shrink-0 items-center gap-2.5 border px-2.5
                text-left text-[13.5px]
                hover:bg-hover hover:text-ink
              `,
              current === section
                ? `border-accent bg-selected font-semibold text-accent-text`
                : `border-transparent text-body`,
              !compact && section === "config" && `mt-auto`,
              compact && `px-3 text-[13px]`,
            ),
          )}
          onClick={() => onNavigate(section)}
        >
          {!compact && <Icon name={section} />}{" "}
          {section.charAt(0).toUpperCase() + section.slice(1)}
        </button>
      ))}
    </nav>
  );
}
