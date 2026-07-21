// Service detail page (`/services/:name`). Shows model info, VRAM
// estimate, placement preview, launch command, and a logs viewer.

import { useMemo, useState } from "react";
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
  IkParams,
  LaunchCommand,
  RuntimeInfo,
  ServingConfig,
  ModelInfo,
  PlacementPreview,
  ServiceDetail,
} from "../../api/client.ts";
import { aggregateBuckets } from "../../api/metrics-aggregate.ts";
import {
  formatBytes,
  formatDuration,
  formatParameterCount,
  formatTimestamp,
  relativeTime,
  serviceProxyUrl,
  metricsWindow,
} from "../../util.ts";
import { Card } from "../ui/Card.tsx";
import { ViewHeader } from "../ui/ViewHeader.tsx";
import { Badge } from "../ui/Badge.tsx";
import { Stat } from "../ui/Stat.tsx";
import { Bar, type BarSegment } from "../ui/Bar.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { CopyButton } from "../ui/CopyButton.tsx";
import { Button } from "../ui/Button.tsx";
import { ButtonLink } from "../ui/ButtonLink.tsx";
import { buttonClassName } from "../ui/buttonStyles.ts";
import { TimeWindowSelect } from "../ui/TimeWindowSelect.tsx";
import {
  TIME_WINDOW_PRESETS,
  windowLabel,
  type TimeWindow,
} from "../ui/timeWindow.ts";
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
  const { t } = useTranslation();
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
        {detail.error?.message ?? t("serviceDetail.notFound")}
      </div>
    );
  }

  const d = detail.data;
  const pending = lifecycle.isPending && lifecycle.variables?.name === name;

  return (
    <div className="flex h-full flex-col">
      {/* Header — fixed height to align with the sidebar wordmark and
          the other views' headers, forming one continuous rule. */}
      <ViewHeader>
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
        <div className="ml-auto flex flex-wrap items-center gap-4">
          <Stat label={t("serviceDetail.port")} value={`:${d.port}`} />
          <Stat label={t("serviceDetail.pid")} value={d.pid ?? "—"} />
          <Stat label={t("serviceDetail.priority")} value={d.priority} />
          <Stat label={t("serviceDetail.lifecycle")} value={d.lifecycle} />
        </div>
      </ViewHeader>

      <div className="flex-1 space-y-4 overflow-auto p-4">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Lifecycle actions */}
          <Card header={t("serviceDetail.actions")} className="lg:col-span-2">
            <div className="flex flex-wrap items-center gap-2">
              <a
                href={serviceProxyUrl(d.port)}
                target="_blank"
                rel="noopener noreferrer"
                className={buttonClassName("blue")}
              >
                <ExternalLinkIcon />
                {t("serviceDetail.open")}
              </a>
              {d.modality !== "embedding" && (
                <ButtonLink
                  variant="iris"
                  to={`/chat?model=${encodeURIComponent(d.name)}`}
                >
                  <ChatIcon className="w-3.5 h-3.5" />
                  {t("serviceDetail.chat")}
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

          {/* Model info */}
          {d.model_info && (
            <Card header={t("serviceDetail.model")}>
              <ModelInfoGrid model={d.model_info} />
            </Card>
          )}

          {/* Configuration */}
          <Card header={t("serviceDetail.configuration")}>
            <ConfigGrid detail={d} />
          </Card>

          {/* Serving (llama-cpp services only) */}
          {d.serving && (
            <Card
              header={t("serviceDetail.serving")}
              bodyClassName="max-h-72 overflow-y-auto p-4"
            >
              <ServingGrid serving={d.serving} runtime={d.runtime ?? null} />
            </Card>
          )}

          {/* Memory estimate */}
          {d.estimate && (
            <Card header={t("serviceDetail.memoryEstimate")}>
              <EstimateGrid
                estimate={d.estimate}
                observedPeakBytes={d.observed_peak_bytes}
              />
            </Card>
          )}
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Placement */}
          <Card header={t("serviceDetail.placement")} className="lg:col-span-2">
            <PlacementSection
              current={d.current_allocation}
              placementOverride={d.placement_override}
              placement={d.placement_preview ?? null}
            />
          </Card>

          {/* Launch command */}
          <Card
            header={t("serviceDetail.launchCommand")}
            className="lg:col-span-2"
          >
            <LaunchCommandSection name={d.name} />
          </Card>
        </div>

        {/* Per-service stats */}
        {d.modality !== "embedding" && <ServiceMetrics name={d.name} />}

        {/* Logs */}
        <Card header={t("serviceDetail.logs")} bodyClassName="p-0">
          <LogsViewer name={d.name} />
        </Card>
      </div>
    </div>
  );
}

function ModelInfoGrid({ model }: { model: ModelInfo }) {
  const { t } = useTranslation();
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {model.model_name && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.name")}</dt>
          <dd className="text-primary">{model.model_name}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.file")}</dt>
      <dd className="flex items-center gap-1">
        <span className="font-mono text-xs text-primary">
          {model.file_name}
        </span>
        <CopyButton value={model.file_name} />
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.architecture")}</dt>
      <dd className="font-mono text-primary">{model.architecture}</dd>
      {model.parameter_count !== undefined &&
        model.parameter_count !== null && (
          <>
            <dt className="text-tertiary">{t("serviceDetail.parameters")}</dt>
            <dd
              className="font-mono text-primary"
              title={`${model.parameter_count.toLocaleString()} parameters`}
            >
              {formatParameterCount(model.parameter_count)}
            </dd>
          </>
        )}
      <dt className="text-tertiary">{t("serviceDetail.onDisk")}</dt>
      <dd className="font-mono text-primary">
        {formatBytes(model.total_tensor_bytes)}
      </dd>
      {model.block_count !== undefined && model.block_count !== null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.layers")}</dt>
          <dd className="font-mono text-primary">{model.block_count}</dd>
        </>
      )}
      {model.trained_context_length !== undefined &&
        model.trained_context_length !== null && (
          <>
            <dt className="text-tertiary">
              {t("serviceDetail.trainedContext")}
            </dt>
            <dd className="font-mono text-primary">
              {t("serviceDetail.tokensValue", {
                value: model.trained_context_length.toLocaleString(),
              })}
            </dd>
          </>
        )}
      {model.shard_count > 1 && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.shards")}</dt>
          <dd className="font-mono text-primary">{model.shard_count}</dd>
        </>
      )}
      {model.license && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.license")}</dt>
          <dd className="font-mono text-xs text-primary">{model.license}</dd>
        </>
      )}
    </dl>
  );
}

function ConfigGrid({ detail }: { detail: ServiceDetail }) {
  const { t } = useTranslation();
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      <dt className="text-tertiary">{t("serviceDetail.template")}</dt>
      <dd className="font-mono text-primary">{detail.template}</dd>
      <dt className="text-tertiary">{t("serviceDetail.context")}</dt>
      <dd className="font-mono text-primary">
        {detail.estimate
          ? t("serviceDetail.tokensValue", {
              value: detail.estimate.configured_context.toLocaleString(),
            })
          : "—"}
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.idleTimeout")}</dt>
      <dd className="font-mono text-primary">
        {detail.lifecycle === "persistent"
          ? t("serviceDetail.neverPersistent")
          : formatDuration(detail.idle_timeout_ms)}
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.lastUsed")}</dt>
      <dd className="font-mono text-primary">
        {detail.last_used_ms != null
          ? `${relativeTime(detail.last_used_ms)} (${formatTimestamp(detail.last_used_ms)})`
          : "—"}
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.runId")}</dt>
      <dd className="font-mono text-primary">{detail.run_id ?? "—"}</dd>
      <dt className="text-tertiary">{t("serviceDetail.privatePort")}</dt>
      <dd className="font-mono text-primary">:{detail.private_port}</dd>
      {detail.rolling_mean != null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.estimatorDrift")}</dt>
          <dd className="font-mono text-primary">
            {detail.rolling_mean.toFixed(3)}×{" "}
            <span className="text-tertiary">
              {t("serviceDetail.samples", { value: detail.rolling_samples })}
            </span>
          </dd>
        </>
      )}
    </dl>
  );
}

// The Serving card: runtime kind (+ik knobs), binary, and the curated
// perf/memory knobs, with derived values (per-slot context, fit
// margins) that no config key or argv flag states directly. Paired
// values share a row to keep the card near the Model card's height;
// the card body scroll-caps as insurance.
function ServingGrid({
  serving,
  runtime,
}: {
  serving: ServingConfig;
  runtime: RuntimeInfo | null;
}) {
  const { t } = useTranslation();
  const flag = (on: boolean) =>
    on ? t("serviceDetail.flagOn") : t("serviceDetail.flagOff");
  const binaryName = serving.binary.split("/").at(-1) ?? serving.binary;
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {runtime && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.runtime")}</dt>
          <dd className="font-mono text-primary">{runtime.kind}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.binary")}</dt>
      <dd className="flex items-center gap-1">
        <span className="font-mono text-xs text-primary">{binaryName}</span>
        <CopyButton value={serving.binary} />
      </dd>
      {runtime?.ik && <IkParamRows ik={runtime.ik} />}
      <dt className="text-tertiary">{t("serviceDetail.kvCache")}</dt>
      <dd className="font-mono text-primary">
        {serving.cache_type_k} / {serving.cache_type_v}
        {serving.flash_attn && <span className="text-tertiary"> · fa</span>}
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.parallelSlots")}</dt>
      <dd className="font-mono text-primary">
        {serving.parallel}
        {serving.kv_unified && (
          <span className="text-tertiary">
            {" "}
            · {t("serviceDetail.kvUnified")}
          </span>
        )}
      </dd>
      {serving.effective_context_per_slot != null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.perSlotContext")}</dt>
          <dd className="font-mono text-primary">
            {t("serviceDetail.tokensValue", {
              value: serving.effective_context_per_slot.toLocaleString(),
            })}
          </dd>
        </>
      )}
      {serving.spec_type && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.specDecode")}</dt>
          <dd className="font-mono text-primary">
            {serving.spec_type}
            {serving.draft_model && (
              <span className="text-tertiary"> · {serving.draft_model}</span>
            )}
          </dd>
        </>
      )}
      {serving.expert_offload !== "off" && (
        <>
          <dt className="text-tertiary">
            {t("serviceDetail.expertOffloadMode")}
          </dt>
          <dd className="font-mono text-primary">{serving.expert_offload}</dd>
        </>
      )}
      {(serving.batch_size != null || serving.ubatch_size != null) && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.batchSizes")}</dt>
          <dd className="font-mono text-primary">
            {serving.batch_size ?? "—"} / {serving.ubatch_size ?? "—"}
          </dd>
        </>
      )}
      {(serving.threads != null || serving.threads_batch != null) && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.threadsRow")}</dt>
          <dd className="font-mono text-primary">
            {serving.threads ?? "—"} / {serving.threads_batch ?? "—"}
          </dd>
        </>
      )}
      {serving.numa && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.numaRow")}</dt>
          <dd className="font-mono text-primary">{serving.numa}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.memoryFlags")}</dt>
      <dd className="font-mono text-primary">
        {flag(serving.mmap)} / {flag(serving.mlock)}
      </dd>
    </dl>
  );
}

// ik_llama.cpp runtime parameters, rendered inside ConfigGrid's <dl>.
// The fit margins are ananke-computed (they appear nowhere in the
// operator's own config), so the dashboard is the one place to read
// them without inspecting a live argv.
function IkParamRows({ ik }: { ik: IkParams }) {
  const { t } = useTranslation();
  const flag = (on: boolean) =>
    on ? t("serviceDetail.flagOn") : t("serviceDetail.flagOff");
  return (
    <>
      {ik.mla !== undefined && ik.mla !== null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.ikMla")}</dt>
          <dd className="font-mono text-primary">{ik.mla}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.ikDsa")}</dt>
      <dd className="font-mono text-primary">{flag(ik.dsa)}</dd>
      <dt className="text-tertiary">{t("serviceDetail.ikFit")}</dt>
      <dd className="font-mono text-primary">{flag(ik.fit)}</dd>
      {ik.fit && ik.fit_margins_mib && ik.fit_margins_mib.length > 0 && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.ikFitMargins")}</dt>
          <dd className="font-mono text-primary">
            {ik.fit_margins_mib.join(" / ")}
          </dd>
        </>
      )}
      {ik.attn_max_batch !== undefined && ik.attn_max_batch !== null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.ikAmb")}</dt>
          <dd className="font-mono text-primary">{ik.attn_max_batch}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.ikRtr")}</dt>
      <dd className="font-mono text-primary">{flag(ik.runtime_repack)}</dd>
    </>
  );
}

function EstimateGrid({
  estimate,
  observedPeakBytes,
}: {
  estimate: EstimateSummary;
  observedPeakBytes: number;
}) {
  const { t } = useTranslation();
  const total =
    estimate.weights_bytes +
    estimate.kv_bytes_for_context +
    estimate.compute_buffer_bytes_per_device;
  return (
    <div className="space-y-2">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-tertiary">{t("serviceDetail.weights")}</dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.weights_bytes)}
        </dd>
        <dt className="text-tertiary">
          {t("serviceDetail.kvAtContext", {
            ctx: estimate.configured_context.toLocaleString(),
          })}
        </dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.kv_bytes_for_context)}
        </dd>
        <dt className="text-tertiary">{t("serviceDetail.computeDev")}</dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.compute_buffer_bytes_per_device)}
        </dd>
        <dt className="text-tertiary">{t("serviceDetail.total")}</dt>
        <dd className="font-mono text-primary">
          {formatBytes(total)}
          {observedPeakBytes > 0 && (
            <span className="text-tertiary">
              {" "}
              {t("serviceDetail.peak", {
                bytes: formatBytes(observedPeakBytes),
              })}
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
  const { t } = useTranslation();
  const hasCurrent = Object.keys(current).length > 0;
  const hasOverride = Object.keys(placementOverride).length > 0;
  const hasPlacement = placement !== null;

  if (!hasCurrent && !hasOverride && !hasPlacement) return null;

  return (
    <div className="space-y-3">
      {hasPlacement && (
        <div>
          <div className="mb-1 flex items-center gap-2">
            <span className="text-xs text-tertiary">
              {t("serviceDetail.preview")}
            </span>
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
              {t("serviceDetail.noPlacementFits")}
            </span>
          )}
          {placement.expert_offload_bytes > 0 && (
            <div className="mt-1.5 text-xs text-tertiary">
              {t("serviceDetail.expertOffload", {
                layers: placement.expert_offload_layers,
                bytes: formatBytes(placement.expert_offload_bytes),
                count: placement.expert_offload_layers,
              })}
            </div>
          )}
        </div>
      )}

      {(hasCurrent || hasOverride) && (
        <div className="grid grid-cols-2 gap-x-4 text-sm">
          {hasCurrent && (
            <div>
              <div className="mb-0.5 text-xs text-tertiary">
                {t("serviceDetail.currentPledge")}
              </div>
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
                title={t("serviceDetail.configuredOverrideTitle")}
              >
                {t("serviceDetail.configuredOverride")}
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
  const { t } = useTranslation();
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
      label: t("serviceDetail.usedByOthers"),
    },
    {
      variant: thisVariant,
      bytes,
      label: t("serviceDetail.thisService"),
    },
    ...(canGrow
      ? [
          {
            variant: thisVariant,
            bytes: max_bytes - bytes,
            label: t("serviceDetail.growthHeadroom"),
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
  const { t } = useTranslation();
  const map: Record<
    PlacementPreview["verdict"],
    { variant: "success" | "warning" | "danger"; label: string }
  > = {
    fits: { variant: "success", label: t("serviceDetail.fitsNow") },
    needs_eviction: {
      variant: "warning",
      label: t("serviceDetail.needsEviction"),
    },
    does_not_fit: { variant: "danger", label: t("serviceDetail.doesNotFit") },
  };
  const { variant, label } = map[verdict];
  return <Badge variant={variant}>{label}</Badge>;
}

function LaunchCommandSection({ name }: { name: string }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const { data, error, isPending } = useServiceCommand(name, open);

  return (
    <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer select-none text-xs text-tertiary hover:text-secondary">
        {t("serviceDetail.expandToCompute")}
      </summary>
      <div className="mt-2">
        {open && isPending && <Spinner />}
        {error && (
          <span className="text-sm text-danger">
            {t("serviceDetail.cannotCompute", { error: error.message })}
          </span>
        )}
        {data && (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <CommandPanel
              label={t("serviceDetail.standalone")}
              command={data.on_empty}
            />
            <CommandPanel
              label={t("serviceDetail.currentConditions")}
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
  const { t } = useTranslation();
  return (
    <div>
      <div className="mb-1 flex items-center gap-2">
        <span className="text-xs text-tertiary">{label}</span>
        {command && (
          <>
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                command.env_inherit
                  ? "bg-emerald-500/15 text-emerald-400"
                  : "bg-amber-500/15 text-amber-400"
              }`}
              title={
                command.env_inherit
                  ? t("serviceDetail.envInheritOnHint")
                  : t("serviceDetail.envInheritOffHint")
              }
            >
              {command.env_inherit
                ? t("serviceDetail.envInheritOn")
                : t("serviceDetail.envInheritOff")}
            </span>
            <CopyButton value={renderCommand(command)} />
          </>
        )}
      </div>
      {command ? (
        <pre className="overflow-x-auto whitespace-pre-wrap break-all rounded-sm bg-base p-2 font-mono text-xs text-primary">
          {renderCommand(command)}
        </pre>
      ) : (
        <div className="flex items-center justify-center rounded-sm bg-base p-4 text-xs text-danger">
          {t("serviceDetail.doesNotFitCurrent")}
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
    <div className="flex flex-wrap items-center gap-2">
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
  if (cmd.env.length > 0) {
    if (cmd.env_inherit) {
      lines.push("# inherits daemon env + overrides below");
    } else {
      lines.push("# clean env — only vars below");
    }
  }
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
  const { t } = useTranslation();
  const [timeWindow, setTimeWindow] = useState<TimeWindow>({
    kind: "relative",
    durationMs: TIME_WINDOW_PRESETS[0]!.durationMs,
  });
  // Freeze `now` per window selection so the query key doesn't churn.
  const { since, until, end, bucket } = useMemo(
    () => metricsWindow(timeWindow),
    [timeWindow],
  );
  const xMin = since / 1000;
  const xMax = end / 1000;

  const metrics = useMetrics({ service: name, since, until, bucket });
  const buckets = aggregateBuckets(metrics.data?.buckets ?? []);

  const totalRequests = buckets.reduce((s, b) => s + b.requestCount, 0);
  const totalErrors = buckets.reduce((s, b) => s + b.errorCount, 0);
  const totalInputTokens = buckets.reduce((s, b) => s + b.promptTokens, 0);
  const totalOutputTokens = buckets.reduce((s, b) => s + b.completionTokens, 0);
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
  const avgEffectiveTps =
    buckets.reduce((s, b) => s + b.totalWeightedEffectiveTps, 0) /
    Math.max(
      1,
      buckets.reduce((s, b) => s + b.effectiveTpsRequests, 0),
    );
  // When no request in the window carries an input/output split (all
  // non-streaming with no engine timings), show the end-to-end effective throughput
  // instead of two zeros.
  const hasSplitTps = buckets.some(
    (b) => b.outputTpsRequests > 0 || b.inputTpsRequests > 0,
  );

  return (
    <div className="space-y-4">
      <Card
        header={t("serviceDetail.stats")}
        headerAction={<TimeWindowSelect onChange={setTimeWindow} />}
      >
        {/* The effective throughput stat is always shown; the input/output
            decode rates are added alongside it when the window has
            split-capable rows. The column count tracks the tile count. */}
        <div
          className={`grid grid-cols-2 gap-4 ${
            hasSplitTps ? "sm:grid-cols-8" : "sm:grid-cols-6"
          }`}
        >
          <Stat
            label={t("serviceDetail.requestsInPeriod", {
              range: windowLabel(timeWindow),
            })}
            value={totalRequests}
          />
          <Stat label={t("serviceDetail.errors")} value={totalErrors} />
          <Stat
            label={t("serviceDetail.inputTokens")}
            value={totalInputTokens.toLocaleString()}
          />
          <Stat
            label={t("serviceDetail.outputTokens")}
            value={totalOutputTokens.toLocaleString()}
          />
          <Stat
            label={t("serviceDetail.avgLatency")}
            value={totalRequests > 0 ? `${Math.round(avgLatency)}ms` : "—"}
          />
          {hasSplitTps && (
            <>
              <Stat
                label={t("serviceDetail.avgTpsIn")}
                value={totalRequests > 0 ? avgInputTps.toFixed(1) : "—"}
              />
              <Stat
                label={t("serviceDetail.avgTpsOut")}
                value={totalRequests > 0 ? avgOutputTps.toFixed(1) : "—"}
              />
            </>
          )}
          <Stat
            label={t("serviceDetail.avgTpsEffective")}
            value={totalRequests > 0 ? avgEffectiveTps.toFixed(1) : "—"}
          />
        </div>
      </Card>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card header={t("stats.requestRate")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[
              buckets.map((b) => b.ts),
              buckets.map((b) => b.requestCount),
            ]}
            series={[
              {
                label: t("stats.requests"),
                stroke: CHART_PALETTE[0],
                fill: "rgba(139,124,248,0.08)",
              },
            ]}
          />
        </Card>
        <Card header={t("stats.tokenThroughput")}>
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
                label: t("stats.tokensIn"),
                stroke: CHART_PALETTE[0],
                fill: "rgba(139,124,248,0.08)",
              },
              {
                label: t("stats.tokensOut"),
                stroke: CHART_PALETTE[1],
                fill: "rgba(69,201,138,0.08)",
              },
            ]}
          />
        </Card>
      </div>
      {/* The end-to-end effective line is always shown; the input/output
          decode rates are overlaid on top of it when available. */}
      <Card header={t("stats.tokensPerSecond")}>
        <Chart
          xMin={xMin}
          xMax={xMax}
          data={[
            buckets.map((b) => b.ts),
            ...(hasSplitTps
              ? [
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
                ]
              : []),
            buckets.map((b) =>
              b.effectiveTpsRequests > 0
                ? b.totalWeightedEffectiveTps / b.effectiveTpsRequests
                : null,
            ),
          ]}
          series={[
            ...(hasSplitTps
              ? [
                  {
                    label: t("stats.tpsIn"),
                    stroke: CHART_PALETTE[0],
                    fill: "rgba(139,124,248,0.08)",
                    unit: "tok/s",
                  },
                  {
                    label: t("stats.tpsOut"),
                    stroke: CHART_PALETTE[1],
                    fill: "rgba(69,201,138,0.08)",
                    unit: "tok/s",
                  },
                ]
              : []),
            {
              label: t("stats.tpsEffective"),
              stroke: CHART_PALETTE[2],
              fill: "rgba(224,168,60,0.08)",
              unit: "tok/s",
            },
          ]}
        />
      </Card>
    </div>
  );
}
