import { useLogs, useServiceDetail } from "../api/hooks.ts";
import type { LogLine } from "../api/client.ts";
import { formatBytes, serviceProxyUrl } from "../util.ts";

// Detail block rendered inline inside the services table (in an extra
// `<td colSpan>` row below the service). Assumes its caller has already
// decided to show it; queries are always enabled.
export function ServiceDetailInline({ name }: { name: string }) {
  const { data: detail, error, isPending } = useServiceDetail(name);
  const { data: logs } = useLogs(name);

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

      <LogsView logs={logs?.logs ?? []} />
    </div>
  );
}

function LogsView({ logs }: { logs: LogLine[] }) {
  if (logs.length === 0) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400">
        No recent logs.
      </div>
    );
  }
  return (
    <div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
        Recent logs (newest first, capped)
      </div>
      <pre className="bg-gray-900 dark:bg-black text-gray-100 dark:text-gray-200 text-xs p-2 rounded max-h-80 overflow-y-auto whitespace-pre-wrap">
        {logs.map((l) => {
          const ts = new Date(l.timestamp_ms).toISOString().slice(11, 23);
          const color =
            l.stream === "stderr" ? "text-red-300" : "text-gray-100";
          return (
            <div key={`${l.run_id}-${l.seq}`} className={color}>
              <span className="text-gray-500 dark:text-gray-600">{ts}</span>{" "}
              {l.line}
            </div>
          );
        })}
      </pre>
    </div>
  );
}
