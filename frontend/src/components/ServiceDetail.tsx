import { useState } from "react";

import { useServiceCommand, useServiceDetail } from "../api/hooks.ts";
import type {
  EstimateSummary,
  LaunchCommand,
  ModelInfo,
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
  // Sum of the three components for a "single-device load" lower
  // bound. The actual per-device split depends on placement (which
  // shows up in `current_allocation` once the service has spawned).
  const estimatedTotal =
    estimate.weights_bytes +
    estimate.kv_bytes_for_context +
    estimate.compute_buffer_bytes_per_device;
  return (
    <section className="mb-3">
      <h3 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
        VRAM estimate
      </h3>
      <dl className="grid grid-cols-[auto_1fr] md:grid-cols-[auto_1fr_auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-gray-500 dark:text-gray-400">Weights</dt>
        <dd>{formatBytes(estimate.weights_bytes)}</dd>
        <dt
          className="text-gray-500 dark:text-gray-400"
          title="kv_per_token × configured context"
        >
          KV cache @ {estimate.configured_context.toLocaleString()}
        </dt>
        <dd>
          {formatBytes(estimate.kv_bytes_for_context)}
          {estimate.kv_per_token > 0 && (
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {" "}
              ({estimate.kv_per_token} B/token)
            </span>
          )}
        </dd>
        <dt
          className="text-gray-500 dark:text-gray-400"
          title="Reserved compute buffer per active device"
        >
          Compute buffer
        </dt>
        <dd>
          {formatBytes(estimate.compute_buffer_bytes_per_device)}{" "}
          <span className="text-xs text-gray-500 dark:text-gray-400">
            per device
          </span>
        </dd>
        <dt
          className="text-gray-500 dark:text-gray-400"
          title="Weights + KV + compute buffer, single-device estimate"
        >
          Estimated total
        </dt>
        <dd className="font-medium">
          {formatBytes(estimatedTotal)}
          {observedPeakBytes > 0 && (
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {" "}
              (observed peak {formatBytes(observedPeakBytes)})
            </span>
          )}
        </dd>
      </dl>
    </section>
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
