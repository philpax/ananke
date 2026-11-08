// The event-type toggles. Any number may be on at once; with none on, or
// with "All types" pressed, every type shows.

import { useTranslation } from "react-i18next";

import { cn } from "../ui/cn.ts";
import {
  EVENT_TYPES,
  eventTypeLabelKey,
  toggleType,
  type EventType,
} from "./eventLog.ts";

export function EventTypeFilter({
  selected,
  onChange,
}: {
  readonly selected: ReadonlySet<EventType>;
  readonly onChange: (types: ReadonlySet<EventType>) => void;
}) {
  const { t } = useTranslation();
  return (
    <div
      role="group"
      aria-label={t("events.typeFilter")}
      className="flex flex-wrap items-center gap-1"
    >
      <FilterButton
        pressed={selected.size === 0}
        onClick={() => onChange(new Set())}
      >
        {t("events.allTypes")}
      </FilterButton>
      {EVENT_TYPES.map((type) => (
        <FilterButton
          key={type}
          pressed={selected.has(type)}
          onClick={() => onChange(toggleType(selected, type))}
        >
          {t(eventTypeLabelKey(type))}
        </FilterButton>
      ))}
    </div>
  );
}

function FilterButton({
  pressed,
  onClick,
  children,
}: {
  readonly pressed: boolean;
  readonly onClick: () => void;
  readonly children: string;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      className={cn(
        "h-control-sm border px-2 text-xs font-medium",
        pressed
          ? "border-accent bg-selected text-accent-text"
          : "border-line text-muted hover:bg-hover hover:text-ink",
      )}
    >
      {children}
    </button>
  );
}
