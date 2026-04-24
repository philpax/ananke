import { useDevices } from "../api/hooks.ts";
import type { DeviceSummary } from "../api/client.ts";
import { formatBytes } from "../util.ts";

export function DevicesPanel() {
  const { data, error, isPending } = useDevices();

  if (isPending)
    return <section className="opacity-60">Loading devices…</section>;
  if (error)
    return (
      <section className="text-red-600 dark:text-red-400">
        Devices: {error.message}
      </section>
    );

  return (
    <section>
      <h2 className="text-lg font-semibold mb-2 dark:text-white">Devices</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {data.map((d) => (
          <DeviceCard key={d.id} device={d} />
        ))}
      </div>
    </section>
  );
}

function DeviceCard({ device }: { device: DeviceSummary }) {
  const used = device.total_bytes - device.free_bytes;
  const usedPct =
    device.total_bytes > 0 ? (used / device.total_bytes) * 100 : 0;

  return (
    <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-800 rounded p-3">
      <div className="flex justify-between items-baseline">
        <div className="font-mono text-sm dark:text-gray-200">{device.id}</div>
        <div className="text-xs text-gray-500 dark:text-gray-400">
          {device.name}
        </div>
      </div>
      <div className="mt-2 h-2 bg-gray-200 dark:bg-gray-800 rounded overflow-hidden">
        <div className="h-full bg-blue-500" style={{ width: `${usedPct}%` }} />
      </div>
      <div className="mt-1 text-xs text-gray-600 dark:text-gray-400">
        {formatBytes(used)} / {formatBytes(device.total_bytes)} used ·{" "}
        {formatBytes(device.free_bytes)} free
      </div>
      {device.reservations.length > 0 && (
        <ul className="mt-2 text-xs space-y-0.5">
          {device.reservations.map((r) => (
            <li key={r.service} className="flex justify-between">
              <span className="font-mono">{r.service}</span>
              <span>
                {formatBytes(r.bytes)}
                {r.elastic ? " (elastic)" : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
