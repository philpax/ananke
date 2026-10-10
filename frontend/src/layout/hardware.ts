import type { Device } from "../api/contract";

export function headerMode(count: number, gpuWidth: number) {
  const rows = count <= 3 ? 1 : count <= 8 ? 2 : count <= 12 ? 3 : 4;
  const columns = Math.ceil(count / rows);
  return {
    kind:
      count > 16 || columns * 140 > gpuWidth
        ? "dense"
        : count <= 3
          ? "full"
          : "compact",
    rows,
    columns,
  } as const;
}

export function deviceMemory(device: Device) {
  const used = Math.max(0, device.total_bytes - device.free_bytes);
  const pledged = device.reservations.reduce((sum, r) => sum + r.bytes, 0);
  const percent =
    device.total_bytes > 0
      ? Math.min(100, (used / device.total_bytes) * 100)
      : 0;
  const pledgePercent =
    device.total_bytes > 0
      ? Math.min(
          100 - percent,
          (Math.max(0, pledged - used) / device.total_bytes) * 100,
        )
      : 0;
  return { used, pledged, percent, pledgePercent };
}
