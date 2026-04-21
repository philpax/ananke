import { useLifecycle, useServices } from "../api/hooks.ts";
import type { ServiceSummary } from "../api/client.ts";
import { serviceProxyUrl } from "../util.ts";

type Props = {
  selected: string | null;
  onSelect: (name: string) => void;
};

export function ServicesTable({ selected, onSelect }: Props) {
  const { data, error, isPending } = useServices();
  const lifecycle = useLifecycle();

  if (isPending)
    return <section className="opacity-60">Loading services…</section>;
  if (error)
    return (
      <section className="text-red-600">Services: {error.message}</section>
    );

  return (
    <section>
      <h2 className="text-lg font-semibold mb-2">Services</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm border border-gray-300">
          <thead className="bg-gray-100 text-left">
            <tr>
              <th className="p-2">Name</th>
              <th className="p-2">State</th>
              <th className="p-2">Lifecycle</th>
              <th className="p-2">Priority</th>
              <th className="p-2">PID</th>
              <th className="p-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {data.map((s) => (
              <ServiceRow
                key={s.name}
                svc={s}
                isSelected={selected === s.name}
                onSelect={() => onSelect(s.name)}
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
        <div className="text-red-600 text-sm mt-2">
          Last action: {lifecycle.error.message}
        </div>
      )}
    </section>
  );
}

type LifecycleAction = "start" | "stop" | "restart" | "enable" | "disable";

function ServiceRow({
  svc,
  isSelected,
  onSelect,
  onLifecycle,
  pending,
}: {
  svc: ServiceSummary;
  isSelected: boolean;
  onSelect: () => void;
  onLifecycle: (action: LifecycleAction) => void;
  pending: boolean;
}) {
  const proxyUrl = serviceProxyUrl(svc.port);
  return (
    <tr
      className={
        "border-t cursor-pointer hover:bg-gray-50 " +
        (isSelected ? "bg-blue-50" : "")
      }
      onClick={onSelect}
    >
      <td className="p-2 font-mono">
        <a
          className="text-blue-700 hover:underline"
          href={proxyUrl}
          target="_blank"
          rel="noreferrer"
          onClick={(e) => e.stopPropagation()}
        >
          {svc.name}
        </a>
        <div className="text-xs text-gray-500">:{svc.port}</div>
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
      className="mr-1 px-2 py-0.5 text-xs border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed"
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

function StateBadge({ state }: { state: string }) {
  const color =
    state === "running"
      ? "bg-green-100 text-green-800"
      : state === "starting"
        ? "bg-yellow-100 text-yellow-800"
        : state === "draining"
          ? "bg-orange-100 text-orange-800"
          : state === "failed"
            ? "bg-red-100 text-red-800"
            : state.startsWith("disabled")
              ? "bg-gray-200 text-gray-800"
              : "bg-gray-100 text-gray-600";
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-xs ${color}`}>
      {state}
    </span>
  );
}
