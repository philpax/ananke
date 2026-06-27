// Dashboard overview — the landing page and primary management view.
// Answers "what's happening right now?" with quick stats, device cards
// with memory sparklines, and a full service list with inline activity
// sparklines. A header time range toggle controls all charts.

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import {
  useDevices,
  useServices,
  useMetrics,
  useDeviceSamples,
  useLifecycle,
} from "../../api/hooks.ts";
import type {
  DeviceSummary,
  ServiceSummary,
  DeviceSampleResponse,
  MetricBucketResponse,
} from "../../api/client.ts";
import { formatBytes, serviceProxyUrl } from "../../util.ts";
import { Card } from "../ui/Card.tsx";
import { Stat } from "../ui/Stat.tsx";
import { Bar, type BarSegment } from "../ui/Bar.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { Badge } from "../ui/Badge.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { CopyButton } from "../ui/CopyButton.tsx";
import { Sparkline } from "../ui/Sparkline.tsx";
import { Chart } from "../ui/Chart.tsx";
import { CHART_PALETTE } from "../ui/chart-palette.ts";

type TimeRange = { label: string; ms: number; bucket: string };
const RANGES: TimeRange[] = [
  { label: "1h", ms: 3_600_000, bucket: "1m" },
  { label: "6h", ms: 6 * 3_600_000, bucket: "5m" },
  { label: "24h", ms: 24 * 3_600_000, bucket: "1h" },
];

type ServiceSparkData = Map<string, { ts: number[]; counts: number[] }>;

function buildServiceSparkline(
  buckets: MetricBucketResponse[],
): ServiceSparkData {
  const byService = new Map<string, Map<number, number>>();
  for (const b of buckets) {
    const name = b.service ?? "unknown";
    let m = byService.get(name);
    if (!m) {
      m = new Map();
      byService.set(name, m);
    }
    const ts = Math.floor(b.bucket_start / 1000);
    m.set(ts, (m.get(ts) ?? 0) + b.request_count);
  }
  const result = new Map<string, { ts: number[]; counts: number[] }>();
  for (const [name, m] of byService) {
    const sorted = [...m.entries()].sort((a, b) => a[0] - b[0]);
    result.set(name, {
      ts: sorted.map((e) => e[0]),
      counts: sorted.map((e) => e[1]),
    });
  }
  return result;
}

export function DashboardView() {
  const { t } = useTranslation();
  const services = useServices();
  const devices = useDevices();
  const lifecycle = useLifecycle();

  const [rangeIdx, setRangeIdx] = useState(0);
  const [since, setSince] = useState(() => Date.now() - RANGES[0].ms);
  const range = RANGES[rangeIdx];
  const xMin = since / 1000;
  const xMax = (since + range.ms) / 1000;

  const metrics = useMetrics({ since, bucket: range.bucket });
  const deviceSamples = useDeviceSamples(undefined, since);

  const serviceSpark = useMemo(
    () => buildServiceSparkline(metrics.data?.buckets ?? []),
    [metrics.data],
  );

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
    <div className="flex h-full flex-col">
      <header className="flex h-14 shrink-0 items-center gap-5 border-b border-border-default px-4">
        <h1 className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-primary">
          {t("dashboard.title")}
        </h1>
        <div className="hidden items-center gap-1.5 text-xs text-tertiary sm:flex">
          <span className="font-mono">{window.location.host}</span>
          <CopyButton value={window.location.host} />
        </div>
        <div className="flex items-center gap-1">
          {RANGES.map((r, i) => (
            <button
              key={r.label}
              onClick={() => {
                setRangeIdx(i);
                setSince(Date.now() - RANGES[i].ms);
              }}
              className={`rounded-sm px-2 py-0.5 text-xs transition-colors ${
                i === rangeIdx
                  ? "bg-elevated text-primary"
                  : "text-tertiary hover:text-secondary"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-5">
          <Stat label={t("dashboard.totalServices")} value={totalCount} />
          <Stat label={t("dashboard.runningServices")} value={runningCount} />
          <Stat
            label={t("dashboard.totalVramInUse")}
            value={formatBytes(totalVramUsed)}
          />
          <Stat
            label={t("dashboard.tokensToday")}
            value={tokensRecent.toLocaleString()}
          />
        </div>
      </header>

      <div className="flex-1 space-y-4 overflow-auto p-4">
        {/* Device cards with memory sparklines */}
        <Card header={t("nav.devices")}>
          {devices.isPending ? (
            <Spinner />
          ) : devices.data ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {devices.data.map((d) => (
                <DeviceCard
                  key={d.id}
                  device={d}
                  samples={deviceSamples.data ?? []}
                  xMin={xMin}
                  xMax={xMax}
                />
              ))}
            </div>
          ) : (
            <span className="text-sm text-danger">
              {devices.error?.message}
            </span>
          )}
        </Card>

        {/* Service list with activity sparklines */}
        <Card header={t("nav.services")} bodyClassName="p-0">
          {services.isPending ? (
            <div className="p-4">
              <Spinner />
            </div>
          ) : services.data ? (
            <div className="divide-y divide-border-default">
              {sortedServices.map((s) => (
                <ServiceRow
                  key={s.name}
                  svc={s}
                  sparkData={serviceSpark.get(s.name)}
                  xMin={xMin}
                  xMax={xMax}
                  pending={
                    lifecycle.isPending && lifecycle.variables?.name === s.name
                  }
                  onAction={(action) =>
                    lifecycle.mutate({ action, name: s.name })
                  }
                />
              ))}
            </div>
          ) : (
            <span className="p-4 text-sm text-danger">
              {services.error?.message}
            </span>
          )}
        </Card>
      </div>
    </div>
  );
}

function DeviceCard({
  device,
  samples,
  xMin,
  xMax,
}: {
  device: DeviceSummary;
  samples: DeviceSampleResponse[];
  xMin: number;
  xMax: number;
}) {
  const total = device.total_bytes;
  const used = total - device.free_bytes;
  const pledged = device.reservations.reduce((sum, r) => sum + r.bytes, 0);
  const pledgedExtra = pledged > used ? pledged - used : 0;

  const segments: BarSegment[] = [
    { variant: "used", bytes: used, label: "used" },
    { variant: "growth", bytes: pledgedExtra, label: "pledged" },
    { variant: "headroom", bytes: Math.max(0, total - used - pledgedExtra) },
  ];

  const chartData = useMemo(() => {
    const filtered = samples
      .filter((s) => s.device === device.id)
      .sort((a, b) => a.timestamp_ms - b.timestamp_ms);
    return [
      filtered.map((s) => Math.floor(s.timestamp_ms / 1000)),
      filtered.map((s) => s.used_bytes / 1e9),
    ] as (number | null)[][];
  }, [samples, device.id]);

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-xs text-primary">{device.id}</span>
        <span className="text-xs text-tertiary">{device.name}</span>
      </div>
      <Chart
        data={chartData}
        series={[
          {
            label: "Used",
            stroke: CHART_PALETTE[0],
            fill: "rgba(139,124,248,0.08)",
            unit: "GB",
          },
        ]}
        height={100}
        xMin={xMin}
        xMax={xMax}
      />
      <Bar total={total} segments={segments} />
      <div className="text-xs text-tertiary">
        {formatBytes(used)} / {formatBytes(total)}
        {pledged > 0 && <> · {formatBytes(pledged)} pledged</>}
      </div>
      {device.reservations.length > 0 && (
        <div className="space-y-0.5">
          {device.reservations.map((r) => (
            <div key={r.service} className="flex items-center gap-2 text-sm">
              <Link
                to={`/services/${encodeURIComponent(r.service)}`}
                className="font-mono text-xs text-accent hover:underline"
              >
                {r.service}
              </Link>
              <span className="font-mono text-xs text-tertiary">
                {formatBytes(r.bytes)}
              </span>
              {r.elastic && <Badge variant="accent">elastic</Badge>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ServiceRow({
  svc,
  sparkData,
  xMin,
  xMax,
  pending,
  onAction,
}: {
  svc: ServiceSummary;
  sparkData?: { ts: number[]; counts: number[] };
  xMin: number;
  xMax: number;
  pending: boolean;
  onAction: (
    action: "start" | "stop" | "restart" | "enable" | "disable",
  ) => void;
}) {
  const { t } = useTranslation();

  const canStart = ["idle", "stopped", "failed", "evicted"].includes(svc.state);
  const canStop = ["running", "starting", "draining"].includes(svc.state);
  const isDisabled = svc.state.startsWith("disabled");

  // Sparkline data — zeroFillGaps (applied inside Sparkline) handles
  // the no-data and single-point cases.
  const sparkLineData: (number | null)[][] = sparkData
    ? [sparkData.ts, sparkData.counts]
    : [[], []];

  return (
    <div className="flex items-center gap-3 px-4 py-2 transition-colors hover:bg-elevated/60">
      <Link
        to={`/services/${encodeURIComponent(svc.name)}`}
        className="flex flex-1 items-center gap-3 overflow-hidden"
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
        <div className="ml-auto flex shrink-0 items-center gap-3">
          <div className="h-6 w-20 shrink-0">
            <Sparkline
              data={sparkLineData}
              color={CHART_PALETTE[0]}
              height={24}
              xMin={xMin}
              xMax={xMax}
            />
          </div>
          <a
            href={serviceProxyUrl(svc.port)}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="rounded-[3px] bg-elevated px-1.5 py-0.5 font-mono text-xs text-accent ring-1 ring-inset ring-border-strong transition-colors hover:bg-border-strong"
          >
            :{svc.port}
          </a>
        </div>
      </Link>
      {/* Sized to hold the three-button case at full size with a tight
          gap, so the right-aligned columns line up across rows however
          many actions a row shows. */}
      <div className="flex w-[110px] shrink-0 items-center justify-end gap-1.5 border-l border-border-default pl-3">
        {canStart && (
          <IconButton
            label={t("services.actions.start")}
            variant="primary"
            onClick={() => onAction("start")}
            disabled={pending}
          >
            <PlayIcon />
          </IconButton>
        )}
        {canStop && (
          <>
            <IconButton
              label={t("services.actions.restart")}
              variant="secondary"
              onClick={() => onAction("restart")}
              disabled={pending}
            >
              <RestartIcon />
            </IconButton>
            <IconButton
              label={t("services.actions.stop")}
              variant="danger"
              onClick={() => onAction("stop")}
              disabled={pending}
            >
              <StopIcon />
            </IconButton>
          </>
        )}
        {isDisabled ? (
          <IconButton
            label={t("services.actions.enable")}
            variant="secondary"
            onClick={() => onAction("enable")}
            disabled={pending}
          >
            <PowerIcon />
          </IconButton>
        ) : (
          <IconButton
            label={t("services.actions.disable")}
            variant="ghost"
            onClick={() => onAction("disable")}
            disabled={pending}
          >
            <DisableIcon />
          </IconButton>
        )}
      </div>
    </div>
  );
}

type IconButtonProps = {
  label: string;
  variant: "primary" | "secondary" | "ghost" | "danger";
  onClick: () => void;
  disabled: boolean;
  children: React.ReactNode;
};

const ICON_VARIANT: Record<IconButtonProps["variant"], string> = {
  primary: "text-accent hover:bg-accent/15",
  secondary: "text-secondary hover:bg-elevated",
  ghost: "text-tertiary hover:bg-elevated hover:text-secondary",
  danger: "text-danger hover:bg-danger/15",
};

function IconButton({
  label,
  variant,
  onClick,
  disabled,
  children,
}: IconButtonProps) {
  return (
    <button
      type="button"
      title={label}
      onClick={(e) => {
        e.stopPropagation();
        e.preventDefault();
        onClick();
      }}
      disabled={disabled}
      className={`inline-flex h-7 w-7 items-center justify-center rounded-md transition-colors disabled:opacity-40 ${ICON_VARIANT[variant]}`}
    >
      {children}
    </button>
  );
}

/* --- Icons --- */

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="1" />
    </svg>
  );
}

function RestartIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

function PowerIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 2v10" />
      <path d="M18.4 6.6a9 9 0 1 1-12.77.04" />
    </svg>
  );
}

function DisableIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M4.93 4.93l14.14 14.14" />
    </svg>
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
