import { useServiceDetail } from "../api/hooks.ts";
import type {
  EstimateSummary,
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
