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
        <table className="min-w-full text-sm border border-gray-300 dark:border-gray-800">
          <thead className="bg-gray-100 dark:bg-gray-900/50 text-left">
            <tr>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700">
                Name
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700">
                State
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700">
                Lifecycle
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700">
                Priority
              </th>
              <th className="p-2 border-b border-gray-300 dark:border-gray-700">
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
        <td className="p-2 font-mono">
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
          <Btn onClick={() => onLifecycle("start")} disabled={pending}>
            Start
          </Btn>
          <Btn onClick={() => onLifecycle("stop")} disabled={pending}>
            Stop
          </Btn>
          <Btn onClick={() => onLifecycle("restart")} disabled={pending}>
            Restart
          </Btn>
          <Btn onClick={() => onLifecycle("enable")} disabled={pending}>
            Enable
          </Btn>
          <Btn onClick={() => onLifecycle("disable")} disabled={pending}>
            Disable
          </Btn>
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
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      className="mr-1 px-2 py-0.5 text-xs border border-gray-300 dark:border-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-700/50 disabled:opacity-40 disabled:cursor-not-allowed"
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
