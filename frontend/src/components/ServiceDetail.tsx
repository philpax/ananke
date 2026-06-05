import { useState } from "react";

import { useServiceCommand, useServiceDetail } from "../api/hooks.ts";
import type {
  DevicePlacement,
  EstimateSummary,
  LaunchCommand,
  ModelInfo,
  PlacementPreview,
  ServiceDetail,
} from "../api/client.ts";
import { formatBytes, formatParameterCount, serviceProxyUrl } from "../util.ts";
import { LogsView } from "./LogsView.tsx";

// Detail block rendered inline inside the services table (in an extra
// `<td colSpan>` row below the service). Assumes its caller has already
// decided to show it; queries are always enabled.
export function ServiceDetailInline({ name }: { name: string }) {
  const { data: detail, error, isPending } = useServiceDetail(name);

  if (isPending) return <div className="opacity-60">Loading {name}…</div>;
  if (error)
    return (
      <div className="text-red-600 dark:text-red-400">
        Detail: {error.message}
      </div>
    );

  const proxyUrl = serviceProxyUrl(detail.port);

  return (
    <div>
      <dl className="grid grid-cols-[auto_1fr] md:grid-cols-[auto_1fr_auto_1fr] gap-x-4 gap-y-1 text-sm mb-3">
        <dt className="text-gray-500 dark:text-gray-400">Template</dt>
        <dd>{detail.template}</dd>
        {detail.modality === "embedding" && (
          <>
            <dt className="text-gray-500 dark:text-gray-400">Modality</dt>
            <dd>
              <span
                className="inline-block px-1.5 py-0.5 text-[10px] rounded bg-teal-100 text-teal-700 dark:bg-teal-950 dark:text-teal-300 align-middle"
                title="Embedding model: serves /v1/embeddings rather than /v1/chat/completions"
              >
                embedding
              </span>
            </dd>
          </>
        )}
        <dt className="text-gray-500 dark:text-gray-400">Port</dt>
        <dd className="break-all">
          <a
            className="text-blue-700 dark:text-blue-300 hover:underline"
            href={proxyUrl}
            target="_blank"
            rel="noreferrer"
          >
            {proxyUrl}
          </a>
          <span className="text-gray-500 dark:text-gray-400">
            {" "}
            (upstream :{detail.private_port})
          </span>
        </dd>
        <dt className="text-gray-500 dark:text-gray-400">Idle timeout</dt>
        <dd className="tabular-nums">{detail.idle_timeout_ms} ms</dd>
        {detail.rolling_mean !== null && detail.rolling_mean !== undefined && (
          <>
            <dt className="text-gray-500 dark:text-gray-400">Rolling mean</dt>
            <dd className="tabular-nums">
              {detail.rolling_mean.toFixed(3)} (n={detail.rolling_samples})
            </dd>
          </>
        )}
        {detail.observed_peak_bytes > 0 && (
          <>
            <dt className="text-gray-500 dark:text-gray-400">Observed peak</dt>
            <dd>{formatBytes(detail.observed_peak_bytes)}</dd>
          </>
        )}
      </dl>

      <ModelSection model={detail.model_info ?? null} />
      <EstimateSection
        estimate={detail.estimate ?? null}
        observedPeakBytes={detail.observed_peak_bytes}
      />
      <AllocationSection
        current={detail.current_allocation}
        placementOverride={detail.placement_override}
      />
      <PlacementNowSection placement={detail.placement_preview ?? null} />

      <LaunchCommandSection name={name} />

      <LogsView name={name} />
    </div>
  );
}

function ModelSection({ model }: { model: ModelInfo | null }) {
  if (!model) return null;
  return (
    <section className="mb-3">
      <h3 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
        Model
      </h3>
      <dl className="grid grid-cols-[auto_1fr] md:grid-cols-[auto_1fr_auto_1fr] gap-x-4 gap-y-1 text-sm">
        {model.model_name && (
          <>
            <dt className="text-gray-500 dark:text-gray-400">Name</dt>
            <dd className="break-all">{model.model_name}</dd>
          </>
        )}
        <dt className="text-gray-500 dark:text-gray-400">File</dt>
        <dd className="font-mono text-xs break-all">{model.file_name}</dd>
        <dt className="text-gray-500 dark:text-gray-400">Architecture</dt>
        <dd className="font-mono">
          {model.architecture}
          {model.has_mmproj && (
            <span
              className="ml-2 inline-block px-1.5 py-0.5 text-[10px] rounded bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300 align-middle"
              title="Multimodal: a vision projector is configured for this service"
            >
              vision
            </span>
          )}
        </dd>
        {model.parameter_count !== null &&
          model.parameter_count !== undefined && (
            <>
              <dt className="text-gray-500 dark:text-gray-400">Parameters</dt>
              <dd
                className="tabular-nums"
                title={`${model.parameter_count.toLocaleString()} parameters`}
              >
                {formatParameterCount(model.parameter_count)}
              </dd>
            </>
          )}
        <dt className="text-gray-500 dark:text-gray-400">On disk</dt>
        <dd>{formatBytes(model.total_tensor_bytes)}</dd>
        {model.block_count !== null && model.block_count !== undefined && (
          <>
            <dt className="text-gray-500 dark:text-gray-400">Layers</dt>
            <dd className="tabular-nums">{model.block_count}</dd>
          </>
        )}
        {model.trained_context_length !== null &&
          model.trained_context_length !== undefined && (
            <>
              <dt className="text-gray-500 dark:text-gray-400">
                Trained context
              </dt>
              <dd className="tabular-nums">
                {model.trained_context_length.toLocaleString()} tokens
              </dd>
            </>
          )}
        {model.shard_count > 1 && (
          <>
            <dt className="text-gray-500 dark:text-gray-400">Shards</dt>
            <dd className="tabular-nums">{model.shard_count}</dd>
          </>
        )}
        {model.license && (
          <>
            <dt className="text-gray-500 dark:text-gray-400">License</dt>
            <dd className="font-mono text-xs">{model.license}</dd>
          </>
        )}
      </dl>
    </section>
  );
}

function EstimateSection({
  estimate,
  observedPeakBytes,
}: {
  estimate: EstimateSummary | null;
  observedPeakBytes: number;
}) {
  if (!estimate) return null;
  // Sum of the three components for a "single-device load" lower bound. The
  // actual per-device split shows up in the placement bars below.
  const estimatedTotal =
    estimate.weights_bytes +
    estimate.kv_bytes_for_context +
    estimate.compute_buffer_bytes_per_device;
  return (
    <section className="mb-3">
      <h3 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
        VRAM estimate
      </h3>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm">
        <span>
          <span className="text-gray-500 dark:text-gray-400">Weights</span>{" "}
          {formatBytes(estimate.weights_bytes)}
        </span>
        <span title="kv_per_token × configured context">
          <span className="text-gray-500 dark:text-gray-400">
            KV @ {estimate.configured_context.toLocaleString()}
          </span>{" "}
          {formatBytes(estimate.kv_bytes_for_context)}
        </span>
        <span title="Reserved compute buffer per active device">
          <span className="text-gray-500 dark:text-gray-400">Compute/dev</span>{" "}
          {formatBytes(estimate.compute_buffer_bytes_per_device)}
        </span>
        <span
          className="font-medium"
          title="Weights + KV + compute buffer, single-device estimate"
        >
          <span className="text-gray-500 dark:text-gray-400 font-normal">
            Total
          </span>{" "}
          {formatBytes(estimatedTotal)}
          {observedPeakBytes > 0 && (
            <span className="text-xs text-gray-500 dark:text-gray-400 font-normal">
              {" "}
              (peak {formatBytes(observedPeakBytes)})
            </span>
          )}
        </span>
      </div>
    </section>
  );
}

// Per-device utilisation bars showing where the service's VRAM would land
// right now and whether it fits without eviction. Placed after the static
// `AllocationSection` (current pledge / configured override) so the live view
// reads below the declared one.
function PlacementNowSection({
  placement,
}: {
  placement: PlacementPreview | null;
}) {
  if (!placement) return null;
  return (
    <section className="mb-3">
      <div className="flex items-center gap-2 mb-1">
        <h3 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
          Placement now
        </h3>
        <FitBadge verdict={placement.verdict} />
      </div>
      {placement.devices.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          {placement.devices.map((d) => (
            <DeviceBar key={d.device} device={d} verdict={placement.verdict} />
          ))}
        </div>
      ) : (
        <div className="text-xs text-gray-500 dark:text-gray-400">
          No placement fits the allowed GPUs.
        </div>
      )}
    </section>
  );
}

// A per-device utilisation bar: what is already in use by other services
// (grey), this service's reserved share (solid blue, or amber when it does not
// fit in currently-free VRAM), and — for a dynamic service — the room it may
// grow into up to its max (a lighter extension), against total capacity.
function DeviceBar({
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
  const pct = (n: number) =>
    hasBar ? Math.min(100, (n / total_bytes) * 100) : 0;
  const othersPct = pct(used_by_others_bytes);
  const thisPct = Math.min(100 - othersPct, pct(bytes));
  const growthPct = canGrow
    ? Math.min(100 - othersPct - thisPct, pct(max_bytes - bytes))
    : 0;
  const thisColor =
    verdict === "needs_eviction"
      ? "bg-amber-500 dark:bg-amber-400"
      : "bg-blue-500 dark:bg-blue-400";
  const growthColor =
    verdict === "needs_eviction"
      ? "bg-amber-300 dark:bg-amber-800"
      : "bg-blue-300 dark:bg-blue-800";
  return (
    <div className="text-xs">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-gray-600 dark:text-gray-300">
          {name}
        </span>
        <span className="tabular-nums text-gray-500 dark:text-gray-400">
          {formatBytes(bytes)}
          {canGrow && <>–{formatBytes(max_bytes)}</>}
          {hasBar && <> / {formatBytes(total_bytes)}</>}
        </span>
      </div>
      {hasBar && (
        // Widths are runtime values, so they go through `style` rather than a
        // (necessarily static) Tailwind width utility.
        <div
          className="mt-0.5 flex h-2 w-full overflow-hidden rounded bg-gray-200 dark:bg-gray-800"
          title={`${formatBytes(used_by_others_bytes)} in use by others · ${formatBytes(bytes)}${canGrow ? ` (up to ${formatBytes(max_bytes)})` : ""} this service · ${formatBytes(total_bytes)} total`}
        >
          <div
            className="bg-gray-400 dark:bg-gray-600"
            style={{ width: `${othersPct}%` }}
          />
          <div className={thisColor} style={{ width: `${thisPct}%` }} />
          {growthPct > 0 && (
            <div className={growthColor} style={{ width: `${growthPct}%` }} />
          )}
        </div>
      )}
    </div>
  );
}

function FitBadge({ verdict }: { verdict: PlacementPreview["verdict"] }) {
  const styles: Record<
    PlacementPreview["verdict"],
    { label: string; cls: string; title: string }
  > = {
    fits: {
      label: "fits now",
      cls: "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
      title: "Starts now in currently-free VRAM — no eviction needed.",
    },
    needs_eviction: {
      label: "needs eviction",
      cls: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
      title:
        "Fits within the hardware, but currently-free VRAM is insufficient — the daemon would reclaim or evict lower-priority services to make room.",
    },
    does_not_fit: {
      label: "does not fit",
      cls: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
      title: "Too large for the allowed GPUs even with everything else gone.",
    },
  };
  const s = styles[verdict];
  return (
    <span
      className={`inline-block px-1.5 py-0.5 text-[10px] rounded ${s.cls}`}
      title={s.title}
    >
      {s.label}
    </span>
  );
}

function AllocationSection({
  current,
  placementOverride,
}: {
  current: ServiceDetail["current_allocation"];
  placementOverride: ServiceDetail["placement_override"];
}) {
  const hasCurrent = Object.keys(current).length > 0;
  const hasOverride = Object.keys(placementOverride).length > 0;
  if (!hasCurrent && !hasOverride) return null;
  return (
    <section className="mb-3">
      <h3 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
        Placement
      </h3>
      <div className="grid grid-cols-2 gap-x-4 text-sm">
        <div>
          <div className="text-gray-500 dark:text-gray-400 mb-0.5">
            Current pledge
          </div>
          {hasCurrent ? (
            <ul className="font-mono text-xs">
              {Object.entries(current).map(([slot, mb]) => (
                <li key={slot} className="tabular-nums">
                  <span className="text-gray-500 dark:text-gray-400">
                    {slot}:
                  </span>{" "}
                  {mb.toLocaleString()} MiB
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-xs text-gray-500 dark:text-gray-400">
              not running
            </div>
          )}
        </div>
        {hasOverride && (
          <div>
            <div
              className="text-gray-500 dark:text-gray-400 mb-0.5"
              title="Manual override declared in config; overrides the packer's per-device split"
            >
              Configured override
            </div>
            <ul className="font-mono text-xs">
              {Object.entries(placementOverride).map(([slot, mb]) => (
                <li key={slot} className="tabular-nums">
                  <span className="text-gray-500 dark:text-gray-400">
                    {slot}:
                  </span>{" "}
                  {mb.toLocaleString()} MiB
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

// The full llama-server / command argv ananke uses (or would use) for this
// service. Collapsed by default: expanding it enables the query, which makes
// the daemon run the estimator + packer on demand rather than on every poll.
function LaunchCommandSection({ name }: { name: string }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const { data, error, isPending } = useServiceCommand(name, open);

  const copy = async (cmd: LaunchCommand) => {
    try {
      await navigator.clipboard.writeText(renderCommand(cmd));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard unavailable (e.g. an insecure context) — leave the
      // button label unchanged; the command is still visible to select.
    }
  };

  return (
    <section className="mb-3">
      <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary className="cursor-pointer text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 select-none">
          Launch command
        </summary>
        <div className="mt-2 text-sm">
          {open && isPending && <div className="opacity-60">Computing…</div>}
          {error && (
            <div className="text-red-600 dark:text-red-400">
              Cannot compute command: {error.message}
            </div>
          )}
          {data && (
            <>
              <div className="flex items-center gap-2 mb-1">
                <SourceBadge source={data.source} />
                <button
                  type="button"
                  onClick={() => void copy(data)}
                  className="px-1.5 py-0.5 text-[10px] rounded border border-gray-300 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800"
                >
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
              <pre className="bg-gray-50 dark:bg-black/30 text-xs p-2 rounded overflow-x-auto whitespace-pre-wrap break-all font-mono">
                {renderCommand(data)}
              </pre>
            </>
          )}
        </div>
      </details>
    </section>
  );
}

function SourceBadge({ source }: { source: LaunchCommand["source"] }) {
  if (source === "running") {
    return (
      <span
        className="inline-block px-1.5 py-0.5 text-[10px] rounded bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300"
        title="The service is running; this is the configuration it was launched with."
      >
        running
      </span>
    );
  }
  return (
    <span
      className="inline-block px-1.5 py-0.5 text-[10px] rounded bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300"
      title="The service is not running; this is what it would launch with on the next start, given the current config and free VRAM."
    >
      preview
    </span>
  );
}

// Render a launch command as a shell-pasteable, multi-line string: env
// assignments first, then the binary, then each flag (paired with its value
// when it takes one), joined with backslash continuations. Every token is
// shell-quoted so JSON/regex arguments stay intact.
function renderCommand(cmd: LaunchCommand): string {
  const lines: string[] = [];
  for (const e of cmd.env) lines.push(`${e.key}=${shellQuote(e.value)}`);
  const [binary, ...rest] = cmd.argv;
  if (binary !== undefined) lines.push(shellQuote(binary));
  for (let i = 0; i < rest.length; i++) {
    const tok = rest[i];
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

// Minimal POSIX shell quoting: leave shell-safe tokens bare, single-quote the
// rest (escaping embedded single quotes the standard `'\''` way).
function shellQuote(s: string): string {
  if (s.length === 0) return "''";
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(s)) return s;
  return `'${s.replace(/'/g, "'\\''")}'`;
}
