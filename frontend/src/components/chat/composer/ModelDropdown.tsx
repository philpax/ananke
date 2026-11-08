// The composer's model picker: a button showing the model's state and
// vision support that opens into a filter field over the chat models.
// Enter picks the first match and Escape closes.

import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import type { ServiceSummary } from "../../../api/client.ts";
import { cn } from "../../ui/cn.ts";
import { fieldVariants } from "../../ui/form/fieldVariants.ts";
import { TextInput } from "../../ui/form/TextInput.tsx";
import { Icon } from "../../ui/icon/Icon.tsx";
import { Chip } from "../../ui/status/Chip.tsx";
import { StatusDot } from "../../ui/status/StatusDot.tsx";

type ModelDropdownProps = {
  readonly models: readonly ServiceSummary[];
  readonly selected: string | null;
  readonly onSelect: (name: string) => void;
  readonly className?: string;
};

export function ModelDropdown({
  models,
  selected,
  onSelect,
  className,
}: ModelDropdownProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  const close = () => {
    setOpen(false);
    setFilter("");
  };
  const pick = (name: string) => {
    onSelect(name);
    close();
  };

  useEffect(() => {
    if (!open) return;
    const handler = (e: Event) => {
      if (e.target instanceof Node && !ref.current?.contains(e.target)) {
        setOpen(false);
        setFilter("");
      }
    };
    document.addEventListener("mousedown", handler);
    document.addEventListener("touchstart", handler);
    return () => {
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("touchstart", handler);
    };
  }, [open]);

  const current = models.find((s) => s.name === selected);
  const needle = filter.toLowerCase();
  const shown = needle
    ? models.filter((s) => s.name.toLowerCase().includes(needle))
    : models;

  return (
    <div ref={ref} className={cn("relative", className)}>
      {open ? (
        <TextInput
          autoFocus
          value={filter}
          aria-label={t("chat.filterModels")}
          placeholder={t("chat.filterModels")}
          onChange={(e) => setFilter(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") close();
            else if (e.key === "Enter" && shown[0]) pick(shown[0].name);
          }}
          className="w-full"
        />
      ) : (
        <button
          type="button"
          aria-haspopup="listbox"
          onClick={() => setOpen(true)}
          className={cn(
            fieldVariants({ size: "md" }),
            "flex w-full items-center gap-2 text-left hover:border-line-hover",
          )}
        >
          {current ? (
            <ModelLabel service={current} />
          ) : (
            <span className="truncate text-faint">
              {selected ?? t("chat.selectModelPlaceholder")}
            </span>
          )}
          <Icon name="chevron" size="3.5" className="ml-auto text-muted" />
        </button>
      )}
      {open && (
        <div className="absolute bottom-full left-0 z-20 mb-1 max-h-72 w-full overflow-y-auto border border-line-strong bg-surface">
          {shown.length === 0 ? (
            <p className="px-3 py-2 text-sm text-muted">
              {t("chat.noMatchingModels")}
            </p>
          ) : (
            shown.map((s) => (
              <button
                key={s.name}
                type="button"
                onClick={() => pick(s.name)}
                className={cn(
                  "flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-hover",
                  s.name === selected && "bg-selected",
                )}
              >
                <ModelLabel service={s} />
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function ModelLabel({ service }: { readonly service: ServiceSummary }) {
  return (
    <>
      <StatusDot state={service.state} reason={service.disabled_reason} />
      <span className="min-w-0 truncate font-mono text-ink">
        {service.name}
      </span>
      {service.has_mmproj && (
        <Chip tone="vision" className="shrink-0">
          vision
        </Chip>
      )}
    </>
  );
}
