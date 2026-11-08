// The strip across the top of every page: who and where the daemon is,
// whether the live connection is up, and every device's memory. On narrow
// screens the cells fold behind a one-line summary. The band owns the top
// safe-area inset, since nothing sits above it.
//
// It does not clip its overflow: a cell's holder list hangs below it over
// the page.

import { useId, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";

import { useDevices, useInfo } from "../../../api/hooks.ts";
import { useLiveStatus } from "../../../api/live/useLive.ts";
import { cn } from "../../ui/cn.ts";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { BandIdentity } from "./BandIdentity.tsx";
import { BandSummary } from "./BandSummary.tsx";
import { DeviceCells } from "./DeviceCells.tsx";
import { bandSummary, formatUptime, splitDevices } from "./hardware.ts";
import { useUptime } from "./useUptime.ts";

export function HardwareBand({ className }: { readonly className?: string }) {
  const { t } = useTranslation();
  const devices = useDevices();
  const info = useInfo();
  const stale = useLiveStatus() === "reconnecting";
  const uptime = useUptime(info.data?.uptime_ms, info.dataUpdatedAt);
  const [expanded, setExpanded] = useState(false);
  const panelId = useId();
  const summaryRef = useRef<HTMLButtonElement>(null);

  const split = splitDevices(devices.data ?? []);
  const host = window.location.host;
  const uptimeText =
    uptime === undefined
      ? undefined
      : t("band.uptime", { value: formatUptime(uptime) });

  function onKeyDown(event: KeyboardEvent) {
    if (event.key !== "Escape" || !expanded) return;
    setExpanded(false);
    summaryRef.current?.focus();
  }

  return (
    <header
      onKeyDown={onKeyDown}
      className={cn(
        "border-b-2 border-line-strong bg-surface pt-[env(safe-area-inset-top)] md:flex md:h-[calc(var(--spacing-band)+env(safe-area-inset-top))] md:items-stretch",
        className,
      )}
    >
      <BandIdentity
        uptime={uptimeText}
        summary={
          <BandSummary
            ref={summaryRef}
            summary={bandSummary(split)}
            expanded={expanded}
            onToggle={() => setExpanded(!expanded)}
            controls={panelId}
          />
        }
      />
      <div
        id={panelId}
        className={cn(
          "max-md:flex max-md:max-h-[44dvh] max-md:flex-col max-md:gap-2.5 max-md:overflow-auto max-md:border-t max-md:border-line max-md:px-3 max-md:pt-2.5 max-md:pb-3 md:flex md:min-w-0 md:flex-1 md:border-l md:border-line",
          !expanded && "max-md:hidden",
        )}
      >
        <p className="flex min-w-0 items-center gap-3 text-xs text-muted md:hidden">
          <span className="inline-flex min-w-0 items-center gap-0.5 sm:hidden">
            <span className="truncate font-mono text-body">{host}</span>
            <CopyButton value={host} />
          </span>
          {uptimeText}
        </p>
        <DeviceCells
          devices={split}
          pending={devices.isPending}
          error={devices.error?.message}
          stale={stale}
        />
      </div>
    </header>
  );
}
