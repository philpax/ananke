// Pure logic behind the hardware band: what a device's memory adds up to,
// how the GPU grid is shaped for a given count, and the one-line summary
// shown on narrow screens. Nothing here measures the page; the cells pick
// their own presentation with container queries.

import type { DeviceReservation, DeviceSummary } from "../../../api/client.ts";
import type { Tone } from "../../ui/tone.ts";

export type DeviceUsage = {
  readonly used: number;
  readonly pledged: number;
  /// What is pledged to services but not yet resident: the bar's second
  /// segment.
  readonly pledgedBeyondUsed: number;
  readonly total: number;
  /// Resident memory as a share of the total, in [0, 100].
  readonly percent: number;
};

export function deviceUsage(device: DeviceSummary): DeviceUsage {
  const total = Math.max(0, device.total_bytes);
  const used = Math.min(total, Math.max(0, total - device.free_bytes));
  const pledged = device.reservations.reduce((sum, r) => sum + r.bytes, 0);
  const percent = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  return {
    used,
    pledged,
    pledgedBeyondUsed: Math.max(0, pledged - used),
    total,
    percent,
  };
}

export const WARN_PERCENT = 80;
export const DANGER_PERCENT = 90;

export function usageTone(percent: number): Tone {
  if (percent >= DANGER_PERCENT) return "danger";
  if (percent >= WARN_PERCENT) return "warn";
  return "neutral";
}

export type GpuGrid = {
  readonly rows: number;
  readonly columns: number;
  /// Past 16 cards the grid is one row of slim cells, whatever the width.
  readonly dense: boolean;
};

export function gpuGrid(count: number): GpuGrid {
  if (count <= 0) return { rows: 0, columns: 0, dense: false };
  if (count > 16) return { rows: 1, columns: count, dense: true };
  const rows = count <= 3 ? 1 : count <= 8 ? 2 : count <= 12 ? 3 : 4;
  return { rows, columns: Math.ceil(count / rows), dense: false };
}

export type BandDevices = {
  readonly gpus: readonly DeviceSummary[];
  readonly cpu: DeviceSummary | undefined;
};

/// GPUs in index order, then host memory. Anything else is ignored.
export function splitDevices(devices: readonly DeviceSummary[]): BandDevices {
  const gpus = devices
    .filter((d) => d.id.startsWith(GPU_PREFIX))
    .toSorted((a, b) => gpuIndex(a.id) - gpuIndex(b.id));
  return { gpus, cpu: devices.find((d) => d.id === CPU_ID) };
}

/// The index alone, for a dense cell's label: "gpu:3" becomes "3".
export function gpuShortId(id: string): string {
  return id.startsWith(GPU_PREFIX) ? id.slice(GPU_PREFIX.length) : id;
}

/// When the busiest GPU is this many points above the mean, the summary
/// appends it: a mean of 45 % over one full card and one empty one hides
/// the card that blocks placement.
export const SUMMARY_MAX_GAP = 20;

export type BandSummary = {
  readonly gpu:
    | {
        readonly mean: number;
        readonly max: number;
        /// The busiest card's tone, not the mean's.
        readonly tone: Tone;
        readonly showMax: boolean;
      }
    | undefined;
  readonly cpu: { readonly percent: number; readonly tone: Tone } | undefined;
};

export function bandSummary({ gpus, cpu }: BandDevices): BandSummary {
  const percents = gpus.map((d) => Math.round(deviceUsage(d).percent));
  const gpu =
    percents.length === 0
      ? undefined
      : summariseGpus(
          Math.round(percents.reduce((a, b) => a + b, 0) / percents.length),
          Math.max(...percents),
        );
  const cpuPercent =
    cpu === undefined ? undefined : Math.round(deviceUsage(cpu).percent);
  return {
    gpu,
    cpu:
      cpuPercent === undefined
        ? undefined
        : { percent: cpuPercent, tone: usageTone(cpuPercent) },
  };
}

/// Drops the vendor and consumer-line prefix: "NVIDIA GeForce RTX 4090"
/// becomes "RTX 4090".
export function shortDeviceName(name: string): string {
  return name.replace(/^NVIDIA\s+(?:GeForce\s+)?/, "").trim() || name;
}

/// Services holding memory on a device, largest first.
export function deviceHolders(
  device: DeviceSummary,
): readonly DeviceReservation[] {
  return device.reservations.toSorted(
    (a, b) => b.bytes - a.bytes || a.service.localeCompare(b.service),
  );
}

/// Two units at most: "59s", "12m", "3h 4m", "40d 2h".
export function formatUptime(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  if (days > 0) return `${days}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m`;
  return `${seconds}s`;
}

const GPU_PREFIX = "gpu:";
const CPU_ID = "cpu";

function gpuIndex(id: string): number {
  const index = Number(id.slice(GPU_PREFIX.length));
  return Number.isFinite(index) ? index : Number.MAX_SAFE_INTEGER;
}

function summariseGpus(mean: number, max: number): BandSummary["gpu"] {
  return {
    mean,
    max,
    tone: usageTone(max),
    showMax: max - mean >= SUMMARY_MAX_GAP,
  };
}
