import clsx from "clsx";
import { useEffect, useRef, useState } from "preact/hooks";

import type { Device, Info } from "../api/contract";
import { Copy } from "../ui/Copy";
import { Icon } from "../ui/Icon";
import { memory, uptime } from "../ui/format";
import { deviceMemory, headerMode } from "./hardware";

export function Hardware({
  devices,
  info,
  compact,
  error,
}: {
  readonly devices: readonly Device[];
  readonly info?: Info;
  readonly compact: boolean;
  readonly error?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const [width, setWidth] = useState(0);
  const gpuRef = useRef<HTMLDivElement>(null);
  const gpus = devices.filter((d) => d.id.startsWith("gpu:"));
  const cpu = devices.find((d) => d.id === "cpu");
  useEffect(() => {
    const el = gpuRef.current;
    if (!el || compact) return;
    const observer = new ResizeObserver((entries) =>
      setWidth(entries[0].contentRect.width),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [compact]);
  const mode = headerMode(gpus.length, width);
  const gpuPercent = gpus.length
    ? Math.round(
        gpus.reduce((s, d) => s + deviceMemory(d).percent, 0) / gpus.length,
      )
    : null;
  const host = location.host;
  const resourceLabel = `gpu ${gpuPercent == null ? "—" : `${gpuPercent}%`} · cpu ${cpu ? `${Math.round(deviceMemory(cpu).percent)}%` : "—"}`;
  return (
    <header
      data-header-mode={compact ? "compact" : "desktop"}
      class={clsx(
        `shrink-0 border-b-2 border-line-strong bg-surface`,
        !compact && `flex h-[88px] items-stretch overflow-hidden`,
      )}
    >
      <div
        class={clsx(
          "flex items-center justify-between",
          compact
            ? `gap-[9px] px-2.5 py-1.5`
            : `w-[316px] shrink-0 gap-3 bg-selected-soft/20 px-3.5 py-2`,
        )}
      >
        <div class="flex shrink-0 items-center gap-2 text-accent">
          <Icon name="mark" size={compact ? 19 : 26} />
          <span
            class={clsx(
              `font-mono leading-none font-bold`,
              compact ? `text-base` : `text-[21px]`,
            )}
          >
            ananke
          </span>
        </div>
        <div
          class={clsx(
            "flex min-w-0",
            compact
              ? "ml-auto items-center"
              : `flex-col items-end justify-center gap-1`,
          )}
        >
          <span class="inline-flex min-w-0 items-center gap-0.5 text-body">
            <span class="truncate font-mono text-[11.5px] leading-none">
              {host}
            </span>
            <Copy text={host} />
          </span>
          {!compact && (
            <span
              class="
                text-[11.5px] leading-none whitespace-nowrap text-muted
                tabular-nums
              "
            >
              {info?.uptime_ms == null
                ? "Uptime unavailable"
                : `up ${uptime(info.uptime_ms)}`}
            </span>
          )}
        </div>
        {compact && (
          <button
            type="button"
            aria-label="Hardware resources"
            aria-expanded={expanded}
            class="
              ml-auto flex h-11 shrink-0 items-center gap-[7px] border
              border-line-strong bg-raised px-2.5 font-mono text-[11.5px]
              text-body
            "
            onClick={() => setExpanded(!expanded)}
          >
            {resourceLabel}
            <span class="text-muted">{expanded ? "▴" : "▾"}</span>
          </button>
        )}
      </div>
      {compact ? (
        expanded && (
          <div
            class="
              flex max-h-[44dvh] flex-col gap-[11px] overflow-auto border-t
              border-line px-3 pt-2.5 pb-3
            "
          >
            <span class="text-xs text-muted">
              {info?.uptime_ms == null
                ? "Uptime unavailable"
                : `up ${uptime(info.uptime_ms)}`}
            </span>
            <div
              aria-label="Device memory"
              class="
                grid grid-cols-[repeat(auto-fit,minmax(148px,1fr))] gap-x-3.5
                gap-y-[9px]
              "
            >
              {devices.map((d) => (
                <DeviceCard key={d.id} device={d} simple />
              ))}
            </div>
            {error && (
              <span role="alert" class="text-xs text-danger">
                {error}
              </span>
            )}
          </div>
        )
      ) : (
        <div
          aria-label="Hardware memory"
          class="
            grid w-[1120px] min-w-0 shrink
            grid-cols-[minmax(0,3fr)_minmax(0,1fr)]
          "
        >
          <div
            ref={gpuRef}
            data-gpu-mode={mode.kind}
            class="grid min-w-0 border-r border-line"
            style={{
              gridTemplateColumns: `repeat(${Math.max(1, mode.kind === "dense" ? gpus.length : mode.columns)},minmax(0,1fr))`,
              gridTemplateRows: `repeat(${mode.kind === "dense" ? 1 : mode.rows},minmax(0,1fr))`,
            }}
          >
            {gpus.map((d, index) => (
              <DeviceCard
                key={d.id}
                device={d}
                dense={mode.kind === "dense"}
                compact={mode.kind === "compact"}
                divider={index % mode.columns !== 0}
                rowDivider={mode.kind === "compact" && index >= mode.columns}
              />
            ))}
          </div>
          <div class="min-w-0 border-r border-line">
            {cpu ? (
              <DeviceCard device={cpu} />
            ) : (
              <p class="p-3 text-xs text-muted">
                {error ?? "CPU data unavailable"}
              </p>
            )}
          </div>
        </div>
      )}
    </header>
  );
}

function DeviceCard({
  device,
  dense = false,
  compact = false,
  divider = false,
  rowDivider = false,
  simple = false,
}: {
  readonly device: Device;
  readonly dense?: boolean;
  readonly compact?: boolean;
  readonly divider?: boolean;
  readonly rowDivider?: boolean;
  readonly simple?: boolean;
}) {
  const m = deviceMemory(device);
  const short = device.name.replace(/^NVIDIA (GeForce )?/, "");
  const label = `${device.id}: ${memory(m.used, 1)} / ${memory(device.total_bytes, 0)} in use · ${memory(m.pledged, 1)} pledged`;
  const color =
    m.percent >= 90
      ? "text-danger"
      : m.percent >= 80
        ? "text-warn"
        : "text-body";
  return (
    <div
      title={label}
      tabIndex={dense ? 0 : undefined}
      class={clsx(
        `relative flex h-full min-w-0 flex-col justify-center`,
        dense
          ? `
            overflow-hidden border-r border-line
            last:border-r-0
          `
          : compact
            ? `gap-1 px-2.5 py-1`
            : simple
              ? `gap-1`
              : `gap-1.5 px-3.5 py-2`,
        divider && !dense && `border-l border-line`,
        rowDivider && `border-t border-line`,
      )}
    >
      {dense ? (
        <>
          <div aria-hidden="true" class="absolute inset-0 flex bg-canvas">
            <div class="bg-accent-data" style={{ width: `${m.percent}%` }} />
            <div
              class="bg-line-strong"
              style={{ width: `${m.pledgePercent}%` }}
            />
          </div>
          <span
            class="
              relative mx-auto font-mono text-[11px] text-ink
              [writing-mode:vertical-rl]
            "
          >
            {device.id} · {Math.round(m.percent)}%
          </span>
          <span class="sr-only">{label}</span>
        </>
      ) : (
        <>
          <div class="flex min-w-0 items-baseline gap-2 text-xs">
            <b class="shrink-0 font-mono">{device.id}</b>
            {!compact && !simple && (
              <span class="truncate text-muted">{short}</span>
            )}
            <b class={clsx(`ml-auto`, color)}>{Math.round(m.percent)}%</b>
          </div>
          <div
            role="img"
            aria-label={label}
            class="flex h-2 shrink-0 border border-line bg-canvas"
          >
            <div class="bg-accent-data" style={{ width: `${m.percent}%` }} />
            <div
              class="bg-line-strong"
              style={{ width: `${m.pledgePercent}%` }}
            />
          </div>
          {!compact && (
            <span class="truncate text-xs text-muted tabular-nums">
              {(m.used / 1024 ** 3).toFixed(1)} /{" "}
              {(device.total_bytes / 1024 ** 3).toFixed(0)} GiB
            </span>
          )}
        </>
      )}
    </div>
  );
}
