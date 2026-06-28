// Service detail page (`/services/:name`). Shows model info, VRAM
// estimate, placement preview, launch command, and a logs viewer.

import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useTranslation } from "react-i18next";

import {
  useServiceDetail,
  useServiceCommand,
  useLifecycle,
  useMetrics,
} from "../../api/hooks.ts";
import type {
  DevicePlacement,
  EstimateSummary,
  LaunchCommand,
  ModelInfo,
  PlacementPreview,
  ServiceDetail,
} from "../../api/client.ts";
import { aggregateBuckets } from "../../api/metrics-aggregate.ts";
import {
  formatBytes,
  formatDuration,
  formatParameterCount,
  serviceProxyUrl,
  RANGES,
} from "../../util.ts";
import { Card } from "../ui/Card.tsx";
import { Badge } from "../ui/Badge.tsx";
import { Stat } from "../ui/Stat.tsx";
import { Bar, type BarSegment } from "../ui/Bar.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { CopyButton } from "../ui/CopyButton.tsx";
import { Button } from "../ui/Button.tsx";
import { ButtonLink } from "../ui/ButtonLink.tsx";
import { buttonClassName } from "../ui/buttonStyles.ts";
import { Chart } from "../ui/Chart.tsx";
import { CHART_PALETTE } from "../ui/chart-palette.ts";
import {
  ChatIcon,
  DisableIcon,
  ExternalLinkIcon,
  PlayIcon,
  PowerIcon,
  RestartIcon,
  StopIcon,
} from "../ui/icons.tsx";
import { LogsViewer } from "../logs/LogsViewer.tsx";

export function ServiceDetailView() {
  const { name } = useParams<{ name: string }>();
  const detail = useServiceDetail(name ?? null);
  const lifecycle = useLifecycle();

  if (detail.isPending) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  if (detail.error || !detail.data) {
    return (
      <div className="p-4 text-sm text-danger">
        {detail.error?.message ?? "service not found"}
      </div>
    );
  }

  const d = detail.data;
  const pending = lifecycle.isPending && lifecycle.variables?.name === name;

  return (
    <div className="flex h-full flex-col">
      {/* Header — fixed height to align with the sidebar wordmark and
          the other views' headers, forming one continuous rule. */}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border-default px-4">
        <Link to="/" className="text-tertiary hover:text-secondary">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
        </Link>
        <StatusDot state={d.state} className="h-2.5 w-2.5" />
        <h1 className="min-w-0 truncate font-mono text-sm font-semibold tracking-[0.02em] text-primary">
          {d.name}
        </h1>
        {d.model_info?.has_mmproj && <Badge variant="vision">vision</Badge>}
        {d.modality === "embedding" && (
          <Badge variant="embedding">embedding</Badge>
        )}
        <div className="ml-auto flex items-center gap-4">
          <Stat label="port" value={`:${d.port}`} />
          <Stat label="pid" value={d.pid ?? "—"} />
          <Stat label="priority" value={d.priority} />
          <Stat label="lifecycle" value={d.lifecycle} />
        </div>
      </header>

      <div className="flex-1 space-y-4 overflow-auto p-4">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Model info */}
          {d.model_info && (
            <Card header="Model">
              <ModelInfoGrid model={d.model_info} />
            </Card>
          )}

          {/* Configuration */}
          <Card header="Configuration">
            <ConfigGrid detail={d} />
          </Card>

          {/* Memory estimate */}
          {d.estimate && (
            <Card header="Memory estimate">
              <EstimateGrid
                estimate={d.estimate}
                observedPeakBytes={d.observed_peak_bytes}
              />
            </Card>
          )}

          {/* Lifecycle actions */}
          <Card header="Actions">
            <div className="flex items-center gap-2">
              <a
                href={serviceProxyUrl(d.port)}
                target="_blank"
                rel="noopener noreferrer"
                className={buttonClassName("blue")}
              >
                <ExternalLinkIcon />
                Open
              </a>
              {d.modality !== "embedding" && (
                <ButtonLink
                  variant="iris"
                  to={`/chat?model=${encodeURIComponent(d.name)}`}
                >
                  <ChatIcon className="w-3.5 h-3.5" />
                  Chat
                </ButtonLink>
              )}
              <LifecycleActions
                state={d.state}
                pending={pending}
                onAction={(action) =>
                  lifecycle.mutate({ action, name: d.name })
                }
              />
            </div>
          </Card>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Placement */}
          <Card header="Placement" className="lg:col-span-2">
            <PlacementSection
              current={d.current_allocation}
              placementOverride={d.placement_override}
              placement={d.placement_preview ?? null}
            />
          </Card>

          {/* Launch command */}
          <Card header="Launch command" className="lg:col-span-2">
            <LaunchCommandSection name={d.name} />
          </Card>
        </div>

        {/* Per-service stats */}
        {d.modality !== "embedding" && <ServiceMetrics name={d.name} />}

        {/* Logs */}
        <Card header="Logs" bodyClassName="p-0">
          <LogsViewer name={d.name} />
        </Card>
      </div>
    </div>
  );
}

function ModelInfoGrid({ model }: { model: ModelInfo }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {model.model_name && (
        <>
          <dt className="text-tertiary">Name</dt>
          <dd className="text-primary">{model.model_name}</dd>
        </>
      )}
      <dt className="text-tertiary">File</dt>
      <dd className="flex items-center gap-1">
        <span className="font-mono text-xs text-primary">
          {model.file_name}
        </span>
        <CopyButton value={model.file_name} />
      </dd>
      <dt className="text-tertiary">Architecture</dt>
      <dd className="font-mono text-primary">{model.architecture}</dd>
      {model.parameter_count !== undefined &&
        model.parameter_count !== null && (
          <>
            <dt className="text-tertiary">Parameters</dt>
            <dd
              className="font-mono text-primary"
              title={`${model.parameter_count.toLocaleString()} parameters`}
            >
              {formatParameterCount(model.parameter_count)}
            </dd>
          </>
        )}
      <dt className="text-tertiary">On disk</dt>
      <dd className="font-mono text-primary">
        {formatBytes(model.total_tensor_bytes)}
      </dd>
      {model.block_count !== undefined && model.block_count !== null && (
        <>
          <dt className="text-tertiary">Layers</dt>
          <dd className="font-mono text-primary">{model.block_count}</dd>
        </>
      )}
      {model.trained_context_length !== undefined &&
        model.trained_context_length !== null && (
          <>
            <dt className="text-tertiary">Trained context</dt>
            <dd className="font-mono text-primary">
              {model.trained_context_length.toLocaleString()} tokens
            </dd>
          </>
        )}
      {model.shard_count > 1 && (
        <>
          <dt className="text-tertiary">Shards</dt>
          <dd className="font-mono text-primary">{model.shard_count}</dd>
        </>
      )}
      {model.license && (
        <>
          <dt className="text-tertiary">License</dt>
          <dd className="font-mono text-xs text-primary">{model.license}</dd>
        </>
      )}
    </dl>
  );
}

function ConfigGrid({ detail }: { detail: ServiceDetail }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      <dt className="text-tertiary">Template</dt>
      <dd className="font-mono text-primary">{detail.template}</dd>
      <dt className="text-tertiary">Context</dt>
      <dd className="font-mono text-primary">
        {detail.estimate
          ? `${detail.estimate.configured_context.toLocaleString()} tokens`
          : "—"}
      </dd>
      <dt className="text-tertiary">Idle timeout</dt>
      <dd className="font-mono text-primary">
        {detail.lifecycle === "persistent"
          ? "never (persistent)"
          : formatDuration(detail.idle_timeout_ms)}
      </dd>
      <dt className="text-tertiary">Run ID</dt>
      <dd className="font-mono text-primary">{detail.run_id ?? "—"}</dd>
      <dt className="text-tertiary">Private port</dt>
      <dd className="font-mono text-primary">:{detail.private_port}</dd>
      {detail.rolling_mean != null && (
        <>
          <dt className="text-tertiary">Estimator drift</dt>
          <dd className="font-mono text-primary">
            {detail.rolling_mean.toFixed(3)}×{" "}
            <span className="text-tertiary">
              ({detail.rolling_samples} samples)
            </span>
          </dd>
        </>
      )}
    </dl>
  );
}

function EstimateGrid({
  estimate,
  observedPeakBytes,
}: {
  estimate: EstimateSummary;
  observedPeakBytes: number;
}) {
  const total =
    estimate.weights_bytes +
    estimate.kv_bytes_for_context +
    estimate.compute_buffer_bytes_per_device;
  return (
    <div className="space-y-2">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-tertiary">Weights</dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.weights_bytes)}
        </dd>
        <dt className="text-tertiary">
          KV @ {estimate.configured_context.toLocaleString()} ctx
        </dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.kv_bytes_for_context)}
        </dd>
        <dt className="text-tertiary">Compute/dev</dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.compute_buffer_bytes_per_device)}
        </dd>
        <dt className="text-tertiary">Total</dt>
        <dd className="font-mono text-primary">
          {formatBytes(total)}
          {observedPeakBytes > 0 && (
            <span className="text-tertiary">
              {" "}
              (peak {formatBytes(observedPeakBytes)})
            </span>
          )}
        </dd>
      </dl>
    </div>
  );
}

function PlacementSection({
  current,
  placementOverride,
  placement,
}: {
  current: ServiceDetail["current_allocation"];
  placementOverride: ServiceDetail["placement_override"];
  placement: PlacementPreview | null;
}) {
  const hasCurrent = Object.keys(current).length > 0;
  const hasOverride = Object.keys(placementOverride).length > 0;
  const hasPlacement = placement !== null;

  if (!hasCurrent && !hasOverride && !hasPlacement) return null;

  return (
    <div className="space-y-3">
      {hasPlacement && (
        <div>
          <div className="mb-1 flex items-center gap-2">
            <span className="text-xs text-tertiary">Preview</span>
            <FitBadge verdict={placement.verdict} />
          </div>
          {placement.devices.length > 0 ? (
            <div className="space-y-1.5">
              {placement.devices.map((dev) => (
                <PlacementBar
                  key={dev.device}
                  device={dev}
                  verdict={placement.verdict}
                />
              ))}
            </div>
          ) : (
            <span className="text-xs text-tertiary">
              No placement fits the allowed GPUs.
            </span>
          )}
          {placement.expert_offload_bytes > 0 && (
            <div className="mt-1.5 text-xs text-tertiary">
              Expert offload: {placement.expert_offload_layers}{" "}
              {placement.expert_offload_layers === 1 ? "layer" : "layers"} ·{" "}
              {formatBytes(placement.expert_offload_bytes)} to CPU
            </div>
          )}
        </div>
      )}

      {(hasCurrent || hasOverride) && (
        <div className="grid grid-cols-2 gap-x-4 text-sm">
          {hasCurrent && (
            <div>
              <div className="mb-0.5 text-xs text-tertiary">Current pledge</div>
              <ul className="font-mono text-xs text-primary">
                {Object.entries(current).map(([slot, mb]) => (
                  <li key={slot}>
                    <span className="text-tertiary">{slot}:</span>{" "}
                    {mb.toLocaleString()} MiB
                  </li>
                ))}
              </ul>
            </div>
          )}
          {hasOverride && (
            <div>
              <div
                className="mb-0.5 text-xs text-tertiary"
                title="Manual override declared in config"
              >
                Configured override
              </div>
              <ul className="font-mono text-xs text-primary">
                {Object.entries(placementOverride).map(([slot, mb]) => (
                  <li key={slot}>
                    <span className="text-tertiary">{slot}:</span>{" "}
                    {mb.toLocaleString()} MiB
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PlacementBar({
  device,
  verdict,
}: {
  device: DevicePlacement;
  verdict: PlacementPreview["verdict"];
}) {
  const {
    device: name,
    bytes,
    max_bytes,
    used_by_others_bytes,
    total_bytes,
  } = device;
  const hasBar = total_bytes > 0;
  const canGrow = max_bytes > bytes;

  const thisVariant: BarSegment["variant"] =
    verdict === "needs_eviction" ? "growth" : "used";

  const segments: BarSegment[] = [
    {
      variant: "reserved",
      bytes: used_by_others_bytes,
      label: "used by others",
    },
    {
      variant: thisVariant,
      bytes,
      label: "this service",
    },
    ...(canGrow
      ? [
          {
            variant: thisVariant,
            bytes: max_bytes - bytes,
            label: "growth headroom",
          } satisfies BarSegment,
        ]
      : []),
    {
      variant: "headroom",
      bytes: Math.max(
        0,
        total_bytes - used_by_others_bytes - Math.max(bytes, max_bytes),
      ),
    },
  ];

  return (
    <div className="text-xs">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-primary">{name}</span>
        <span className="font-mono text-tertiary">
          {formatBytes(bytes)}
          {canGrow && <>&ndash;{formatBytes(max_bytes)}</>}
          {hasBar && <> / {formatBytes(total_bytes)}</>}
        </span>
      </div>
      {hasBar && (
        <Bar total={total_bytes} segments={segments} className="mt-0.5" />
      )}
    </div>
  );
}

function FitBadge({ verdict }: { verdict: PlacementPreview["verdict"] }) {
  const map: Record<
    PlacementPreview["verdict"],
    { variant: "success" | "warning" | "danger"; label: string }
  > = {
    fits: { variant: "success", label: "fits now" },
    needs_eviction: { variant: "warning", label: "needs eviction" },
    does_not_fit: { variant: "danger", label: "does not fit" },
  };
  const { variant, label } = map[verdict];
  return <Badge variant={variant}>{label}</Badge>;
}

function LaunchCommandSection({ name }: { name: string }) {
  const [open, setOpen] = useState(false);
  const { data, error, isPending } = useServiceCommand(name, open);

  return (
    <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer select-none text-xs text-tertiary hover:text-secondary">
        Expand to compute launch command
      </summary>
      <div className="mt-2">
        {open && isPending && <Spinner />}
        {error && (
          <span className="text-sm text-danger">
            Cannot compute: {error.message}
          </span>
        )}
        {data && (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <CommandPanel label="Standalone" command={data.on_empty} />
            <CommandPanel
              label="Current conditions"
              command={data.active ?? null}
            />
          </div>
        )}
      </div>
    </details>
  );
}

function CommandPanel({
  label,
  command,
}: {
  label: string;
  command: LaunchCommand | null;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center gap-2">
        <span className="text-xs text-tertiary">{label}</span>
        {command && <CopyButton value={renderCommand(command)} />}
      </div>
      {command ? (
        <pre className="overflow-x-auto whitespace-pre-wrap break-all rounded-sm bg-base p-2 font-mono text-xs text-primary">
          {renderCommand(command)}
        </pre>
      ) : (
        <div className="flex items-center justify-center rounded-sm bg-base p-4 text-xs text-danger">
          Does not fit under current conditions
        </div>
      )}
    </div>
  );
}

function LifecycleActions({
  state,
  pending,
  onAction,
}: {
  state: string;
  pending: boolean;
  onAction: (
    action: "start" | "stop" | "restart" | "enable" | "disable",
  ) => void;
}) {
  const { t } = useTranslation();
  const canStart = ["idle", "stopped", "failed", "evicted"].includes(state);
  const canStop = ["running", "starting", "draining"].includes(state);
  const isDisabled = state.startsWith("disabled");

  return (
    <div className="flex items-center gap-2">
      {canStart && (
        <Button
          variant="green"
          onClick={() => onAction("start")}
          disabled={pending}
        >
          <PlayIcon />
          {t("services.actions.start")}
        </Button>
      )}
      {canStop && (
        <>
          <Button
            variant="red"
            onClick={() => onAction("stop")}
            disabled={pending}
          >
            <StopIcon />
            {t("services.actions.stop")}
          </Button>
          <Button
            variant="cyan"
            onClick={() => onAction("restart")}
            disabled={pending}
          >
            <RestartIcon />
            {t("services.actions.restart")}
          </Button>
        </>
      )}
      {isDisabled ? (
        <Button
          variant="orange"
          onClick={() => onAction("enable")}
          disabled={pending}
        >
          <PowerIcon />
          {t("services.actions.enable")}
        </Button>
      ) : (
        <Button
          variant="magenta"
          onClick={() => onAction("disable")}
          disabled={pending}
        >
          <DisableIcon />
          {t("services.actions.disable")}
        </Button>
      )}
    </div>
  );
}

function renderCommand(cmd: LaunchCommand): string {
  const lines: string[] = [];
  for (const e of cmd.env) lines.push(`${e.key}=${shellQuote(e.value)}`);
  const [binary, ...rest] = cmd.argv;
  if (binary !== undefined) lines.push(shellQuote(binary));
  for (let i = 0; i < rest.length; i++) {
    const tok = rest[i]!;
    const next = rest[i + 1];
    if (tok.startsWith("-") && next !== undefined && !next.startsWith("-")) {
      lines.push(`  ${shellQuote(tok)} ${shellQuote(next)}`);
      i++;
    } else {
      lines.push(`  ${shellQuote(tok)}`);
    }
  }
  return lines.join(" \\\n");
}

function shellQuote(s: string): string {
  if (s.length === 0) return "''";
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(s)) return s;
  return `'${s.replace(/'/g, "'\\''")}'`;
}

function ServiceMetrics({ name }: { name: string }) {
  const [rangeIdx, setRangeIdx] = useState(0);
  const [since, setSince] = useState(() => Date.now() - RANGES[0].ms);
  const range = RANGES[rangeIdx];
  const xMin = since / 1000;
  const xMax = (since + range.ms) / 1000;

  const metrics = useMetrics({ service: name, since, bucket: range.bucket });
  const buckets = aggregateBuckets(metrics.data?.buckets ?? []);

  const totalRequests = buckets.reduce((s, b) => s + b.requestCount, 0);
  const totalErrors = buckets.reduce((s, b) => s + b.errorCount, 0);
  const totalTokens = buckets.reduce(
    (s, b) => s + b.promptTokens + b.completionTokens,
    0,
  );
  const avgLatency =
    buckets.reduce((s, b) => s + b.totalDurationMs, 0) /
    Math.max(
      1,
      buckets.reduce((s, b) => s + b.timedRequests, 0),
    );
  const avgOutputTps =
    buckets.reduce((s, b) => s + b.totalWeightedOutputTps, 0) /
    Math.max(
      1,
      buckets.reduce((s, b) => s + b.outputTpsRequests, 0),
    );
  const avgInputTps =
    buckets.reduce((s, b) => s + b.totalWeightedInputTps, 0) /
    Math.max(
      1,
      buckets.reduce((s, b) => s + b.inputTpsRequests, 0),
    );

  return (
    <div className="space-y-4">
      <Card
        header="Stats"
        headerAction={
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
        }
      >
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-6">
          <Stat label={`Requests (${range.label})`} value={totalRequests} />
          <Stat label="Errors" value={totalErrors} />
          <Stat label="Tokens" value={totalTokens.toLocaleString()} />
          <Stat
            label="Avg latency"
            value={totalRequests > 0 ? `${Math.round(avgLatency)}ms` : "—"}
          />
          <Stat
            label="Avg in TPS"
            value={totalRequests > 0 ? avgInputTps.toFixed(1) : "—"}
          />
          <Stat
            label="Avg out TPS"
            value={totalRequests > 0 ? avgOutputTps.toFixed(1) : "—"}
          />
        </div>
      </Card>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card header="Request rate">
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[
              buckets.map((b) => b.ts),
              buckets.map((b) => b.requestCount),
            ]}
            series={[
              {
                label: "Requests",
                stroke: CHART_PALETTE[0],
                fill: "rgba(139,124,248,0.08)",
              },
            ]}
          />
        </Card>
        <Card header="Token throughput">
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[
              buckets.map((b) => b.ts),
              buckets.map((b) => b.promptTokens),
              buckets.map((b) => b.completionTokens),
            ]}
            series={[
              {
                label: "Prompt",
                stroke: CHART_PALETTE[0],
                fill: "rgba(139,124,248,0.08)",
              },
              {
                label: "Completion",
                stroke: CHART_PALETTE[1],
                fill: "rgba(69,201,138,0.08)",
              },
            ]}
          />
        </Card>
      </div>
      <Card header="Tokens per second">
        <Chart
          xMin={xMin}
          xMax={xMax}
          data={[
            buckets.map((b) => b.ts),
            buckets.map((b) =>
              b.inputTpsRequests > 0
                ? b.totalWeightedInputTps / b.inputTpsRequests
                : null,
            ),
            buckets.map((b) =>
              b.outputTpsRequests > 0
                ? b.totalWeightedOutputTps / b.outputTpsRequests
                : null,
            ),
          ]}
          series={[
            {
              label: "Input",
              stroke: CHART_PALETTE[0],
              fill: "rgba(139,124,248,0.08)",
              unit: "tok/s",
            },
            {
              label: "Output",
              stroke: CHART_PALETTE[1],
              fill: "rgba(69,201,138,0.08)",
              unit: "tok/s",
            },
          ]}
        />
      </Card>
    </div>
  );
}
