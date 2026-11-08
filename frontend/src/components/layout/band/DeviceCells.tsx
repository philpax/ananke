// The band's device cells: the GPU grid and the host-memory cell beside
// it. The grid's shape is data (`gpuGrid`), passed to CSS as custom
// properties so the class names stay static. On narrow screens both
// wrappers dissolve (`contents`) and every cell joins one auto-fit grid.

import { useTranslation } from "react-i18next";

import { cn } from "../../ui/cn.ts";
import { Message } from "../../ui/feedback/Message.tsx";
import { DeviceCell } from "./DeviceCell.tsx";
import { gpuGrid, type BandDevices } from "./hardware.ts";

type DeviceCellsProps = {
  readonly devices: BandDevices;
  readonly pending: boolean;
  readonly error: string | undefined;
  /// The live connection is down, so these are the last known figures.
  readonly stale: boolean;
};

export function DeviceCells({
  devices,
  pending,
  error,
  stale,
}: DeviceCellsProps) {
  const { t } = useTranslation();
  const { gpus, cpu } = devices;
  const empty = gpus.length === 0 && cpu === undefined;
  if (empty) {
    return (
      <div className="flex items-center md:px-3.5">
        {pending ? (
          <Message kind="loading" />
        ) : error !== undefined ? (
          <Message kind="error" text={error} />
        ) : (
          <Message kind="empty" text={t("band.noDevices")} />
        )}
      </div>
    );
  }

  const grid = gpuGrid(gpus.length);
  const fillers = grid.rows * grid.columns - gpus.length;
  const gridVars: Record<string, number> = {
    "--gpu-cols": grid.columns,
    "--gpu-rows": grid.rows,
  };
  return (
    <div
      role="group"
      aria-label={t("band.cells")}
      className={cn(
        "grid gap-2 max-md:grid-cols-[repeat(auto-fit,minmax(9.25rem,1fr))] md:size-full md:max-w-band-cells md:grid-cols-[minmax(0,3fr)_minmax(0,1fr)] md:gap-px md:border-r md:border-line md:bg-line",
        stale && "opacity-60",
      )}
    >
      <div
        data-rows={grid.rows}
        data-stacked={grid.rows >= 2 ? "" : undefined}
        data-tight={grid.rows >= 3 ? "" : undefined}
        style={gridVars}
        className="max-md:contents md:grid md:min-w-0 md:grid-cols-[repeat(var(--gpu-cols),minmax(0,1fr))] md:grid-rows-[repeat(var(--gpu-rows),minmax(0,1fr))] md:gap-px"
      >
        {gpus.length === 0 ? (
          <p className="flex items-center bg-surface px-3.5 text-xs text-muted max-md:hidden">
            {t("band.noGpus")}
          </p>
        ) : (
          gpus.map((d) => <DeviceCell key={d.id} device={d} kind="gpu" />)
        )}
        {Array.from({ length: fillers }, (_, i) => (
          <div
            key={i}
            aria-hidden="true"
            className="bg-surface max-md:hidden"
          />
        ))}
      </div>
      <div data-rows={1} className="max-md:contents md:grid md:min-w-0">
        {cpu === undefined ? (
          <div aria-hidden="true" className="bg-surface max-md:hidden" />
        ) : (
          <DeviceCell device={cpu} kind="cpu" alignEnd />
        )}
      </div>
    </div>
  );
}
