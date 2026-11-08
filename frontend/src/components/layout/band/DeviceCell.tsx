// One device's memory in the band. The cell is its own container and picks
// its presentation from its own width: full (id, name, percent, bar, GiB),
// compact (id, percent, bar), or dense, where the bar fills the cell and
// the label shrinks to "0 · 85%", turning vertical in a single-row grid.
//
// Focusing or hovering the cell shows which services hold memory on the
// device. On narrow screens, where the cells sit in a scrolling panel that
// would clip a popover, that list is shown in place instead.

import { useId } from "react";
import { useTranslation } from "react-i18next";

import type { DeviceSummary } from "../../../api/client.ts";
import { formatBytes } from "../../../util.ts";
import { cn } from "../../ui/cn.ts";
import { MemoryBar } from "../../ui/data/MemoryBar.tsx";
import { Chip } from "../../ui/status/Chip.tsx";
import { toneText } from "../../ui/tone.ts";
import {
  deviceHolders,
  deviceUsage,
  gpuShortId,
  shortDeviceName,
  usageTone,
} from "./hardware.ts";

type DeviceCellProps = {
  readonly device: DeviceSummary;
  readonly kind: "gpu" | "cpu";
  /// Anchors the popover to the cell's right edge, for the last column.
  readonly alignEnd?: boolean;
};

export function DeviceCell({
  device,
  kind,
  alignEnd = false,
}: DeviceCellProps) {
  const { t } = useTranslation();
  const tipId = useId();
  const usage = deviceUsage(device);
  const percent = Math.round(usage.percent);
  const tone = toneText({ tone: usageTone(usage.percent) });
  const name =
    kind === "gpu" ? shortDeviceName(device.name) : t("band.hostMemory");
  const holders = deviceHolders(device);
  const usageText = t("band.usage", {
    percent,
    used: formatBytes(usage.used),
    total: formatBytes(usage.total),
    pledged: formatBytes(usage.pledgedBeyondUsed),
  });

  return (
    <div
      tabIndex={0}
      role="group"
      aria-label={`${device.id} ${name}`}
      aria-describedby={tipId}
      className="group/device @container/cell relative min-w-0 bg-surface -outline-offset-2"
    >
      <div className="relative flex h-full min-w-0 flex-col justify-center gap-1 overflow-hidden px-2.5 py-1 md:in-data-tight:flex-row md:in-data-tight:items-center md:in-data-tight:gap-2 md:in-data-tight:py-0.5 @max-cell-dense/cell:items-center @max-cell-dense/cell:p-0 @cell-full/cell:gap-1.5 @cell-full/cell:px-3.5 @cell-full/cell:py-2 md:in-data-stacked:@cell-full/cell:gap-1 md:in-data-stacked:@cell-full/cell:py-1 md:in-data-tight:@cell-full/cell:py-0.5">
        <div
          aria-hidden="true"
          className="flex min-w-0 items-baseline gap-2 text-xs md:in-data-tight:shrink-0 @max-cell-dense/cell:hidden"
        >
          <span className="shrink-0 font-mono font-bold text-ink">
            {device.id}
          </span>
          <span className="hidden truncate text-muted @cell-full/cell:inline">
            {name}
          </span>
          <span className={cn("ml-auto font-semibold tabular-nums", tone)}>
            {percent}%
          </span>
        </div>
        <MemoryBar
          variant="device"
          total={usage.total}
          used={usage.used}
          pledged={usage.pledgedBeyondUsed}
          label={usageText}
          className="shrink-0 md:in-data-tight:flex-1 @max-cell-dense/cell:absolute @max-cell-dense/cell:inset-0 @max-cell-dense/cell:h-auto @max-cell-dense/cell:border-0"
        />
        <span
          aria-hidden="true"
          className="hidden truncate text-xs text-muted tabular-nums @cell-full/cell:block md:in-data-stacked:@cell-full/cell:hidden"
        >
          {t("band.gib", {
            used: (usage.used / GIB).toFixed(1),
            total: (usage.total / GIB).toFixed(0),
          })}
        </span>
        <span
          aria-hidden="true"
          className="relative hidden bg-surface/85 px-0.5 font-mono text-2xs text-ink in-data-[rows=1]:[writing-mode:vertical-rl] @max-cell-dense/cell:block"
        >
          {kind === "gpu" ? gpuShortId(device.id) : device.id} · {percent}%
        </span>
      </div>
      <div
        id={tipId}
        role="tooltip"
        className={cn(
          "text-xs text-body max-md:pt-1 md:invisible md:absolute md:top-full md:z-20 md:mt-0.5 md:w-64 md:max-w-[calc(100vw-2rem)] md:border md:border-line-strong md:bg-raised md:px-3 md:py-2 md:group-hover/device:visible md:group-focus-visible/device:visible",
          alignEnd ? "md:right-0" : "md:left-0",
        )}
      >
        <p className="font-semibold text-ink max-md:sr-only">
          {device.id} · {device.name}
        </p>
        <p className="text-muted max-md:sr-only">{usageText}</p>
        {holders.length === 0 ? (
          <p className="text-muted md:mt-1.5">{t("band.noHolders")}</p>
        ) : (
          <ul aria-label={t("band.holders")} className="md:mt-1.5">
            {holders.map((r) => (
              <li key={r.service} className="flex min-w-0 items-center gap-2">
                <span className="truncate font-mono">{r.service}</span>
                {r.elastic && <Chip tone="accent">{t("band.elastic")}</Chip>}
                <span className="ml-auto font-mono text-muted tabular-nums">
                  {formatBytes(r.bytes)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

const GIB = 1024 ** 3;
