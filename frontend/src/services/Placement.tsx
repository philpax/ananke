import type { Detail, Device, Service } from "../api/contract";
import { memory } from "../ui/format";

export function Placement({
  detail,
  summary,
  devices,
}: {
  readonly detail: Detail;
  readonly summary?: Service;
  readonly devices: readonly Device[];
}) {
  const preview = detail.placement_preview;
  const allocations = Object.entries(detail.current_allocation);
  const footprints = allocations.length
    ? allocations.map(([device, mb]) => ({ device, bytes: mb * 1024 ** 2 }))
    : preview?.devices.length
      ? preview.devices
      : (summary?.footprint_devices ??
        Object.entries(detail.placement_override).map(([device, mb]) => ({
          device,
          bytes: mb * 1024 ** 2,
        })));
  const ordered = [...footprints].sort((a, b) =>
    a.device === "cpu"
      ? 1
      : b.device === "cpu"
        ? -1
        : a.device.localeCompare(b.device, undefined, { numeric: true }),
  );
  return (
    <section class="min-w-0">
      <div class="flex flex-wrap items-center gap-3.5 pb-[9px]">
        <h2 class="text-[13px] leading-[1.35] font-semibold">Placement</h2>
        <span class="flex items-center gap-1.5 text-xs text-body">
          <span class="size-[11px] bg-line-strong" aria-hidden="true" />
          other services
        </span>
        <span class="flex items-center gap-1.5 text-xs text-body">
          <span class="size-[11px] bg-accent-data" aria-hidden="true" />
          this service
        </span>
      </div>
      <div class="flex flex-col gap-3 pt-1">
        {ordered.map((p) => {
          const device = devices.find((d) => d.id === p.device);
          const total = device?.total_bytes ?? 0;
          const calculated = preview?.devices.find(
            (d) => d.device === p.device,
          );
          const others =
            calculated?.used_by_others_bytes ??
            device?.reservations
              .filter((r) => r.service !== detail.name)
              .reduce((s, r) => s + r.bytes, 0) ??
            0;
          const base = total ? Math.min(100, (others / total) * 100) : 0;
          const share = total
            ? Math.min(100 - base, (p.bytes / total) * 100)
            : 0;
          const label = `${p.device}: ${memory(p.bytes)} of this service, ${memory(others)} of other services, ${memory(total)} total`;
          return (
            <div key={p.device} class="flex flex-col gap-[5px]">
              <div class="flex items-baseline gap-2.5 text-[12.5px]">
                <span class="font-mono">{p.device}</span>
                <span class="ml-auto text-body tabular-nums">
                  {memory(p.bytes).replace(" GiB", "")} / {memory(total)}
                </span>
              </div>
              <div
                role="img"
                aria-label={label}
                title={label}
                class="flex h-3 border border-line-strong bg-canvas"
              >
                <div class="bg-line-strong" style={{ width: `${base}%` }} />
                <div class="bg-accent-data" style={{ width: `${share}%` }} />
              </div>
            </div>
          );
        })}
      </div>
      {!ordered.length && (
        <p class="py-2 text-[13px] text-muted">Placement is unavailable.</p>
      )}
      {preview?.verdict.kind === "does_not_fit" && (
        <p class="mt-2 text-xs text-danger">
          Does not fit
          {preview.verdict.shortfalls
            .map(
              (s) =>
                ` · ${s.device} needs ${memory(s.requested_bytes)}, ${memory(s.available_bytes)} available`,
            )
            .join("")}
        </p>
      )}
      {preview?.verdict.kind === "needs_eviction" && (
        <p class="mt-2 text-xs text-warn">Placement needs eviction.</p>
      )}
    </section>
  );
}
