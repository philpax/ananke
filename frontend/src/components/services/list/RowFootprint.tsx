// Where a service's memory would land: the total, and per device in that
// device's series colour. A service that cannot be placed reports what it
// needs rather than what it holds, and says so.

import { useTranslation } from "react-i18next";

import type { ServiceSummary } from "../../../api/client.ts";
import { formatBytes } from "../../../util.ts";
import { deviceSeries, type SeriesColor } from "../../ui/chart/series.ts";

export function RowFootprint({
  service,
}: {
  readonly service: ServiceSummary;
}) {
  const { t } = useTranslation();
  if (service.footprint_bytes == null) return null;
  const needed = service.fit_verdict?.kind === "does_not_fit";
  const devices = service.footprint_devices ?? [];
  return (
    <span className="flex min-w-0 flex-wrap items-baseline gap-x-2 font-mono text-xs text-soft">
      <span className="shrink-0">
        {formatBytes(service.footprint_bytes)}
        {needed && ` ${t("services.footprintNeeded")}`}
      </span>
      {devices.length > 0 && (
        <span className="flex min-w-0 flex-wrap gap-x-2 @max-list-spark/list:hidden">
          {devices.map((d) => (
            <span key={d.device} className="shrink-0">
              <span className={DEVICE_TEXT[deviceSeries(d.device)]}>
                {d.device.replace(":", "")}
              </span>{" "}
              {formatBytes(d.bytes)}
            </span>
          ))}
        </span>
      )}
    </span>
  );
}

// Literal classes so Tailwind generates each one.
const DEVICE_TEXT: Readonly<Record<SeriesColor, string>> = {
  "series-1": "text-series-1",
  "series-2": "text-series-2",
  "series-3": "text-series-3",
  "series-4": "text-series-4",
  "series-5": "text-series-5",
  "series-6": "text-series-6",
  "device-cpu": "text-device-cpu",
  ok: "text-ok",
  warn: "text-warn",
  danger: "text-danger",
};
