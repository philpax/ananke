// Dashboard overview — the landing page. Answers "what's happening
// right now?" with device summaries, service cards, recent events,
// and quick stats.

import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { useDevices, useServices, useMetrics } from "../../api/hooks.ts";
import type { DeviceSummary, ServiceSummary } from "../../api/client.ts";
import { formatBytes } from "../../util.ts";
import { Card } from "../ui/Card.tsx";
import { Stat } from "../ui/Stat.tsx";
import { Bar, type BarSegment } from "../ui/Bar.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { Badge } from "../ui/Badge.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { CopyButton } from "../ui/CopyButton.tsx";

export function DashboardView() {
  const { t } = useTranslation();
  const services = useServices();
  const devices = useDevices();

  // Let the daemon default to "last 1h" by omitting since/until.
  // The refetch interval (30s) keeps this fresh.
  const metrics = useMetrics({ bucket: "1h" });

  const runningCount =
    services.data?.filter((s) => s.state === "running").length ?? 0;
  const totalCount = services.data?.length ?? 0;

  const totalVramUsed =
    devices.data?.reduce((sum, d) => {
      if (d.id.startsWith("cpu")) return sum;
      return sum + (d.total_bytes - d.free_bytes);
    }, 0) ?? 0;

  const tokensRecent =
    metrics.data?.buckets.reduce(
      (sum, b) => sum + b.prompt_tokens + b.completion_tokens,
      0,
    ) ?? 0;

  const sortedServices = services.data
    ? [...services.data].sort((a, b) => {
        const rankDiff = stateRank(a.state) - stateRank(b.state);
        if (rankDiff !== 0) return rankDiff;
        return a.name.localeCompare(b.name);
      })
    : [];

  return (
    <div className="space-y-3 p-3">
      {/* Header strip */}
      <div className="flex items-center gap-4 border-b border-border-default pb-3">
        <h1 className="text-base font-medium text-primary">
          {t("dashboard.title")}
        </h1>
        <div className="flex items-center gap-1.5 text-xs text-tertiary">
          <span className="font-mono">{window.location.host}</span>
          <CopyButton value={window.location.host} />
        </div>
        <div className="ml-auto flex items-center gap-4">
          <Stat label={t("dashboard.totalServices")} value={totalCount} />
          <Stat label={t("dashboard.runningServices")} value={runningCount} />
          <Stat
            label={t("dashboard.totalVramInUse")}
            value={formatBytes(totalVramUsed)}
          />
          {tokensRecent > 0 && (
            <Stat
              label={t("dashboard.tokensToday")}
              value={tokensRecent.toLocaleString()}
            />
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        {/* Device summary */}
        <Card header={t("nav.devices")} className="lg:col-span-1">
          {devices.isPending ? (
            <Spinner />
          ) : devices.data ? (
            <div className="space-y-2">
              {devices.data
                .filter((d) => !d.id.startsWith("cpu"))
                .map((d) => (
                  <DeviceMiniCard key={d.id} device={d} />
                ))}
              {devices.data
                .filter((d) => d.id.startsWith("cpu"))
                .map((d) => (
                  <DeviceMiniCard key={d.id} device={d} />
                ))}
            </div>
          ) : (
            <span className="text-sm text-danger">
              {devices.error?.message}
            </span>
          )}
        </Card>

        {/* Service grid */}
        <Card
          header={t("nav.services")}
          className="lg:col-span-2"
          bodyClassName="p-0"
        >
          {services.isPending ? (
            <div className="p-3">
              <Spinner />
            </div>
          ) : services.data ? (
            <div className="divide-y divide-border-default">
              {sortedServices.map((s) => (
                <ServiceMiniRow key={s.name} svc={s} />
              ))}
            </div>
          ) : (
            <span className="text-sm text-danger p-3">
              {services.error?.message}
            </span>
          )}
        </Card>
      </div>
    </div>
  );
}

function DeviceMiniCard({ device }: { device: DeviceSummary }) {
  const total = device.total_bytes;
  const used = total - device.free_bytes;
  const pledged = device.reservations.reduce((sum, r) => sum + r.bytes, 0);
  const pledgedExtra = pledged > used ? pledged - used : 0;

  const segments: BarSegment[] = [
    { variant: "used", bytes: used, label: "used" },
    { variant: "growth", bytes: pledgedExtra, label: "pledged" },
    { variant: "headroom", bytes: Math.max(0, total - used - pledgedExtra) },
  ];

  return (
    <Link to="/devices" className="block">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-xs text-primary">{device.id}</span>
        <span className="text-xs text-tertiary">{device.name}</span>
      </div>
      <Bar total={total} segments={segments} className="mt-1" />
      <div className="mt-1 text-xs text-tertiary">
        {formatBytes(used)} / {formatBytes(total)}
        {pledged > 0 && <> · {formatBytes(pledged)} pledged</>}
      </div>
    </Link>
  );
}

function ServiceMiniRow({ svc }: { svc: ServiceSummary }) {
  return (
    <Link
      to={`/services/${encodeURIComponent(svc.name)}`}
      className="flex items-center gap-3 px-3 py-1.5 hover:bg-elevated transition-colors"
    >
      <StatusDot state={svc.state} />
      <span className="font-mono text-sm text-primary">{svc.name}</span>
      {svc.has_mmproj && <Badge variant="vision">vision</Badge>}
      {svc.modality === "embedding" && (
        <Badge variant="embedding">embedding</Badge>
      )}
      {(svc.inflight_count ?? 0) > 0 && (
        <Badge variant="accent">{svc.inflight_count} in-flight</Badge>
      )}
      <span className="ml-auto font-mono text-xs text-tertiary">
        :{svc.port}
      </span>
    </Link>
  );
}

function stateRank(state: string): number {
  if (state === "running") return 0;
  if (state === "starting") return 1;
  if (state === "draining") return 2;
  if (state === "idle") return 3;
  if (state === "evicted") return 4;
  if (state === "stopped") return 5;
  if (state === "failed") return 6;
  if (state.startsWith("disabled")) return 7;
  return 8;
}
