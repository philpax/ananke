import { useState } from "react";

import { useLifecycle, useServices } from "../api/hooks.ts";
import type { ServiceSummary } from "../api/client.ts";
import { serviceProxyUrl } from "../util.ts";
import { ServiceDetailInline } from "./ServiceDetail.tsx";

const COLUMN_COUNT = 6;

export function ServicesTable() {
  const { data, error, isPending } = useServices();
  const lifecycle = useLifecycle();

  if (isPending)
    return <section className="opacity-60">Loading services…</section>;
  if (error)
    return (
      <section className="text-red-600 dark:text-red-400">
        Services: {error.message}
      </section>
    );

  const sorted = [...data].sort((a, b) => {
    const rankDiff = stateRank(a.state) - stateRank(b.state);
    if (rankDiff !== 0) return rankDiff;
    return a.name.localeCompare(b.name);
  });

  return (
    <section>
      <h2 className="text-lg font-semibold mb-2 dark:text-white">Services</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm border border-gray-300 dark:border-gray-800 table-fixed">
          <thead className="bg-gray-100 dark:bg-gray-900/50 text-left">
            <tr>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700 w-1/4">
                Name
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700 w-20">
                State
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700 w-32">
                Lifecycle
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700 w-20">
                Priority
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700 w-24">
                PID
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => (
              <ServiceRow
                key={s.name}
                svc={s}
                onLifecycle={(action) =>
                  lifecycle.mutate({ action, name: s.name })
                }
                pending={
                  lifecycle.isPending && lifecycle.variables?.name === s.name
                }
              />
            ))}
          </tbody>
        </table>
      </div>
      {lifecycle.error && (
        <div className="text-red-600 dark:text-red-400 text-sm mt-2">
          Last action: {lifecycle.error.message}
        </div>
      )}
    </section>
  );
}

type LifecycleAction = "start" | "stop" | "restart" | "enable" | "disable";

function ServiceRow({
  svc,
  onLifecycle,
  pending,
}: {
  svc: ServiceSummary;
  onLifecycle: (action: LifecycleAction) => void;
  pending: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const proxyUrl = serviceProxyUrl(svc.port);

  return (
    <>
      <tr
        className={`cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 ${
          expanded
            ? "bg-blue-50 dark:bg-blue-900/20"
            : "border-b border-gray-200 dark:border-gray-800"
        }`}
        onClick={() => setExpanded((v) => !v)}
      >
        <td className="p-2 font-mono truncate">
          <span className="inline-block w-3 text-gray-400 dark:text-gray-500 text-xs">
            {expanded ? "▾" : "▸"}
          </span>{" "}
          <a
            className="text-blue-700 dark:text-blue-400 hover:underline"
            href={proxyUrl}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => e.stopPropagation()}
          >
            {svc.name}
          </a>
          <div className="text-xs text-gray-500 dark:text-gray-400 pl-4">
            :{svc.port}
          </div>
        </td>
        <td className="p-2">
          <StateBadge state={svc.state} />
        </td>
        <td className="p-2">{svc.lifecycle}</td>
        <td className="p-2 tabular-nums">{svc.priority}</td>
        <td className="p-2 tabular-nums text-xs">{svc.pid ?? "—"}</td>
        <td className="p-2 whitespace-nowrap">
          {["idle", "stopped", "failed", "evicted"].includes(svc.state) && (
            <Btn
              variant="start"
              onClick={() => onLifecycle("start")}
              disabled={pending}
            >
              Start
            </Btn>
          )}
          {["running", "starting", "draining"].includes(svc.state) && (
            <>
              <Btn
                variant="stop"
                onClick={() => onLifecycle("stop")}
                disabled={pending}
              >
                Stop
              </Btn>
              <Btn
                variant="restart"
                onClick={() => onLifecycle("restart")}
                disabled={pending}
              >
                Restart
              </Btn>
            </>
          )}
          {svc.state.startsWith("disabled") && (
            <Btn
              variant="enable"
              onClick={() => onLifecycle("enable")}
              disabled={pending}
            >
              Enable
            </Btn>
          )}
          {!svc.state.startsWith("disabled") && (
            <Btn
              variant="disable"
              onClick={() => onLifecycle("disable")}
              disabled={pending}
            >
              Disable
            </Btn>
          )}
        </td>
      </tr>
      {expanded && (
        <tr className="bg-blue-50/50 dark:bg-blue-900/10 border-b border-gray-200 dark:border-gray-800">
          <td colSpan={COLUMN_COUNT} className="p-3">
            <ServiceDetailInline name={svc.name} />
          </td>
        </tr>
      )}
    </>
  );
}

function Btn({
  children,
  onClick,
  disabled,
  variant = "default",
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled: boolean;
  variant?: "start" | "stop" | "restart" | "enable" | "disable" | "default";
}) {
  const styles = {
    default:
      "border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700/50",
    start:
      "border-green-300 dark:border-green-800 text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-900/20 hover:bg-green-100 dark:hover:bg-green-900/40",
    stop: "border-red-300 dark:border-red-800 text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-900/20 hover:bg-red-100 dark:hover:bg-red-900/40",
    restart:
      "border-orange-300 dark:border-orange-800 text-orange-700 dark:text-orange-400 bg-orange-50 dark:bg-orange-900/20 hover:bg-orange-100 dark:hover:bg-orange-900/40",
    enable:
      "border-blue-300 dark:border-blue-800 text-blue-700 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 hover:bg-blue-100 dark:hover:bg-blue-900/40",
    disable:
      "border-gray-400 dark:border-gray-600 text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700",
  };

  return (
    <button
      className={`mr-1 px-2 py-0.5 text-xs border rounded disabled:opacity-40 disabled:cursor-not-allowed ${styles[variant]}`}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

// Sort key for table rows. Active states float to the top (a user watching
// the dashboard cares most about what's running now); in-transit states
// follow; idle is mid-pack; terminal / operator-intervention states sink.
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

function StateBadge({ state }: { state: string }) {
  const color =
    state === "running"
      ? "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-400"
      : state === "starting"
        ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-400"
        : state === "draining"
          ? "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-400"
          : state === "failed"
            ? "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-400"
            : state.startsWith("disabled")
              ? "bg-gray-200 text-gray-800 dark:bg-gray-800 dark:text-gray-400"
              : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300";
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-xs ${color}`}>
      {state}
    </span>
  );
}
