// Oneshot job management (`/oneshots`). Lists active and historical
// oneshot jobs, with a submit form for creating new command-template
// jobs and a detail panel showing status + captured logs.

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  useOneshots,
  useCreateOneshot,
  useDeleteOneshot,
} from "../../api/hooks.ts";
import type { OneshotStatus, OneshotRequest } from "../../api/client.ts";
import { Card } from "../ui/Card.tsx";
import { Badge } from "../ui/Badge.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { EmptyState } from "../ui/EmptyState.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { LogsViewer } from "../logs/LogsViewer.tsx";

type OneshotFormState = {
  name: string;
  command: string;
  workdir: string;
  allocationMode: "static" | "dynamic";
  vramGb: string;
  minVramGb: string;
  maxVramGb: string;
  placement: "gpu-only" | "cpu-only" | "hybrid";
  priority: number;
  ttl: string;
  port: string;
  healthPath: string;
  healthTimeout: string;
};

const EMPTY_FORM: OneshotFormState = {
  name: "",
  command: "",
  workdir: "",
  allocationMode: "static",
  vramGb: "4",
  minVramGb: "2",
  maxVramGb: "8",
  placement: "gpu-only",
  priority: 50,
  ttl: "1h",
  port: "",
  healthPath: "",
  healthTimeout: "3m",
};

export function OneshotsView() {
  const { t } = useTranslation();
  const oneshots = useOneshots();
  const createMut = useCreateOneshot();
  const deleteMut = useDeleteOneshot();
  const [showForm, setShowForm] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const sorted = useMemo(() => {
    if (!oneshots.data) return [];
    return [...oneshots.data].sort(
      (a, b) => b.submitted_at_ms - a.submitted_at_ms,
    );
  }, [oneshots.data]);

  const selected = sorted.find((o) => o.id === selectedId) ?? null;

  function handleSubmit(form: OneshotFormState) {
    const argv = form.command.trim().split(/\s+/).filter(Boolean);
    if (argv.length === 0) return;

    const req: OneshotRequest = {
      template: "command",
      command: argv,
      name: form.name.trim() || null,
      workdir: form.workdir.trim() || null,
      port: form.port ? Number(form.port) : null,
      priority: form.priority,
      ttl: form.ttl.trim() || null,
      allocation:
        form.allocationMode === "static"
          ? { mode: "static", vram_gb: Number(form.vramGb) }
          : {
              mode: "dynamic",
              min_vram_gb: Number(form.minVramGb),
              max_vram_gb: Number(form.maxVramGb),
            },
      devices: { placement: form.placement },
      health: form.healthPath.trim()
        ? {
            http: form.healthPath.trim(),
            timeout: form.healthTimeout.trim() || null,
          }
        : null,
    };

    createMut.mutate(req, {
      onSuccess: (resp) => {
        setSelectedId(resp.id);
        setShowForm(false);
      },
    });
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border-default px-4">
        <h1 className="eyebrow !text-primary">{t("nav.oneshots")}</h1>
        <button
          type="button"
          onClick={() => setShowForm((s) => !s)}
          className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-[var(--color-base)] transition-colors hover:bg-accent/90"
        >
          {showForm ? "Cancel" : "New oneshot"}
        </button>
        {oneshots.data && (
          <span className="ml-auto font-mono text-xs text-tertiary">
            {oneshots.data.length} total
          </span>
        )}
      </header>

      <div className="flex-1 overflow-auto p-4">
        {showForm && (
          <OneshotForm
            onSubmit={handleSubmit}
            isPending={createMut.isPending}
            error={createMut.error?.message}
          />
        )}

        {oneshots.isPending && !oneshots.data ? (
          <div className="flex h-full items-center justify-center">
            <Spinner />
          </div>
        ) : sorted.length > 0 ? (
          <div className="space-y-4">
            <Card header="Jobs" bodyClassName="p-0">
              <div className="divide-y divide-border-default">
                {sorted.map((o) => (
                  <OneshotRow
                    key={o.id}
                    oneshot={o}
                    selected={o.id === selectedId}
                    onSelect={() => setSelectedId(o.id)}
                    onDelete={() => {
                      deleteMut.mutate(o.id);
                      if (selectedId === o.id) setSelectedId(null);
                    }}
                    deletePending={
                      deleteMut.isPending && deleteMut.variables === o.id
                    }
                  />
                ))}
              </div>
            </Card>

            {selected && <OneshotDetail oneshot={selected} />}
          </div>
        ) : (
          <EmptyState message="No oneshot jobs. Create one to get started." />
        )}
      </div>
    </div>
  );
}

function OneshotRow({
  oneshot,
  selected,
  onSelect,
  onDelete,
  deletePending,
}: {
  oneshot: OneshotStatus;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  deletePending: boolean;
}) {
  const isTerminal = oneshot.state === "ended" || oneshot.state === "evicted";

  return (
    <div
      className={`flex cursor-pointer items-center gap-3 px-4 py-2 transition-colors hover:bg-elevated/60 ${
        selected ? "bg-elevated/40" : ""
      }`}
      onClick={onSelect}
    >
      <StatusDot state={oneshot.state === "running" ? "running" : "idle"} />
      <span className="font-mono text-sm text-primary">{oneshot.name}</span>
      {oneshot.state === "running" && <Badge variant="success">running</Badge>}
      {oneshot.state === "ended" && (
        <Badge variant={oneshot.exit_code === 0 ? "neutral" : "danger"}>
          {oneshot.exit_code != null ? `exit ${oneshot.exit_code}` : "ended"}
        </Badge>
      )}
      <span className="ml-auto shrink-0 font-mono text-xs text-tertiary">
        :{oneshot.port}
      </span>
      <span className="shrink-0 font-mono text-xs text-tertiary">
        {relativeTime(oneshot.submitted_at_ms)}
      </span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        disabled={deletePending || isTerminal}
        title="Kill"
        className="inline-flex h-7 w-7 items-center justify-center rounded-md text-danger transition-colors hover:bg-danger/15 disabled:opacity-30"
      >
        <KillIcon />
      </button>
    </div>
  );
}

function OneshotDetail({ oneshot }: { oneshot: OneshotStatus }) {
  const submitted = new Date(oneshot.submitted_at_ms).toLocaleString();
  const started = oneshot.started_at_ms
    ? new Date(oneshot.started_at_ms).toLocaleString()
    : null;
  const ended = oneshot.ended_at_ms
    ? new Date(oneshot.ended_at_ms).toLocaleString()
    : null;
  const duration =
    oneshot.started_at_ms && oneshot.ended_at_ms
      ? formatDurationShort(oneshot.ended_at_ms - oneshot.started_at_ms)
      : null;

  return (
    <Card header={oneshot.name} bodyClassName="p-0">
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 border-b border-border-default px-4 py-3 text-sm sm:grid-cols-4">
        <DetailField label="ID" value={oneshot.id} mono />
        <DetailField label="Port" value={`:${oneshot.port}`} mono />
        <DetailField label="State" value={oneshot.state} />
        <DetailField
          label="Exit"
          value={oneshot.exit_code != null ? String(oneshot.exit_code) : "—"}
          mono
        />
        <DetailField label="Submitted" value={submitted} />
        {started && <DetailField label="Started" value={started} />}
        {ended && <DetailField label="Ended" value={ended} />}
        {duration && <DetailField label="Duration" value={duration} mono />}
      </div>
      <LogsViewer name={oneshot.id} />
    </Card>
  );
}

function DetailField({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="text-xs text-tertiary">{label}</dt>
      <dd className={`text-primary ${mono ? "font-mono text-xs" : "text-sm"}`}>
        {value}
      </dd>
    </div>
  );
}

function OneshotForm({
  onSubmit,
  isPending,
  error,
}: {
  onSubmit: (form: OneshotFormState) => void;
  isPending: boolean;
  error?: string;
}) {
  const [form, setForm] = useState<OneshotFormState>(EMPTY_FORM);

  function update<K extends keyof OneshotFormState>(
    key: K,
    value: OneshotFormState[K],
  ) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  const canSubmit = form.command.trim().length > 0 && !isPending;

  return (
    <Card header="New oneshot" className="mb-4">
      <div className="space-y-3">
        {/* Template — only command is available */}
        <FormField label="Template">
          <div className="flex items-center gap-2">
            <span className="rounded-sm bg-elevated px-2.5 py-1 text-xs text-primary ring-1 ring-inset ring-border-strong">
              command
            </span>
            <span
              title="llama-cpp oneshots are not yet supported by the backend"
              className="cursor-not-allowed rounded-sm bg-base px-2.5 py-1 text-xs text-tertiary ring-1 ring-inset ring-border-default line-through"
            >
              llama-cpp
            </span>
          </div>
        </FormField>

        <FormField label="Command" hint="Binary + arguments, space-separated">
          <input
            type="text"
            value={form.command}
            onChange={(e) => update("command", e.target.value)}
            placeholder="comfyui --listen 0.0.0.0 --port 8188"
            className="w-full rounded-sm border border-border-default bg-base px-2 py-1 font-mono text-xs text-primary focus:border-accent focus:outline-none"
          />
        </FormField>

        <div className="grid grid-cols-2 gap-3">
          <FormField label="Name" hint="Optional; auto-generated if blank">
            <input
              type="text"
              value={form.name}
              onChange={(e) => update("name", e.target.value)}
              placeholder="comfyui-test"
              className="w-full rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
            />
          </FormField>
          <FormField label="Working directory">
            <input
              type="text"
              value={form.workdir}
              onChange={(e) => update("workdir", e.target.value)}
              placeholder="/workspace/comfyui"
              className="w-full rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
            />
          </FormField>
        </div>

        {/* Memory, placement, and TTL in one row */}
        <div className="grid grid-cols-3 gap-3">
          <FormField
            label={
              <span className="flex items-center gap-2">
                Memory
                <span className="flex items-center gap-0.5">
                  {(["static", "dynamic"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => update("allocationMode", m)}
                      className={`rounded-sm px-1.5 py-0.5 text-[0.6875rem] transition-colors ${
                        form.allocationMode === m
                          ? "bg-elevated text-primary ring-1 ring-inset ring-border-strong"
                          : "text-tertiary hover:text-secondary"
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </span>
              </span>
            }
            hint="GB"
          >
            {form.allocationMode === "static" ? (
              <input
                type="number"
                step="0.5"
                value={form.vramGb}
                onChange={(e) => update("vramGb", e.target.value)}
                className="w-full rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
              />
            ) : (
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  step="0.5"
                  value={form.minVramGb}
                  onChange={(e) => update("minVramGb", e.target.value)}
                  className="w-full rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
                />
                <span className="text-xs text-tertiary">to</span>
                <input
                  type="number"
                  step="0.5"
                  value={form.maxVramGb}
                  onChange={(e) => update("maxVramGb", e.target.value)}
                  className="w-full rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
                />
              </div>
            )}
          </FormField>
          <FormField label="Placement">
            <select
              value={form.placement}
              onChange={(e) =>
                update(
                  "placement",
                  e.target.value as OneshotFormState["placement"],
                )
              }
              className="w-full rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
            >
              <option value="gpu-only">gpu-only</option>
              <option value="cpu-only">cpu-only</option>
              <option value="hybrid">hybrid</option>
            </select>
          </FormField>
          <FormField label="TTL" hint="e.g. 2h, 30m">
            <input
              type="text"
              value={form.ttl}
              onChange={(e) => update("ttl", e.target.value)}
              className="w-full rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
            />
          </FormField>
        </div>

        {/* Priority slider */}
        <FormField label="Priority" hint="Higher = harder to evict">
          <div className="flex items-center gap-3">
            <input
              type="range"
              min="0"
              max="100"
              value={form.priority}
              onChange={(e) => update("priority", Number(e.target.value))}
              className="flex-1 accent-accent"
            />
            <span className="w-8 shrink-0 text-right font-mono text-xs text-primary">
              {form.priority}
            </span>
          </div>
        </FormField>

        {/* Port + health check combined row */}
        <div className="grid grid-cols-2 gap-3">
          <FormField label="Port" hint="Optional; daemon assigns if blank">
            <input
              type="number"
              value={form.port}
              onChange={(e) => update("port", e.target.value)}
              placeholder="auto"
              className="w-full rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
            />
          </FormField>
          <FormField
            label="Health check"
            hint="Optional path; e.g. /system_stats"
          >
            <div className="flex items-center gap-1">
              <input
                type="text"
                value={form.healthPath}
                onChange={(e) => update("healthPath", e.target.value)}
                placeholder="/system_stats"
                className="flex-1 rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
              />
              {form.healthPath.trim() && (
                <input
                  type="text"
                  value={form.healthTimeout}
                  onChange={(e) => update("healthTimeout", e.target.value)}
                  placeholder="3m"
                  className="w-16 rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary focus:border-accent focus:outline-none"
                />
              )}
            </div>
          </FormField>
        </div>

        {error && (
          <div className="rounded-sm bg-danger/10 px-3 py-1.5 text-xs text-danger">
            {error}
          </div>
        )}

        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => onSubmit(form)}
            disabled={!canSubmit}
            className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-[var(--color-base)] transition-colors hover:bg-accent/90 disabled:opacity-40"
          >
            {isPending ? "Creating…" : "Create oneshot"}
          </button>
        </div>
      </div>
    </Card>
  );
}

function FormField({
  label,
  hint,
  children,
}: {
  label: React.ReactNode;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="text-xs font-medium text-secondary">{label}</label>
      <div className="mt-0.5">{children}</div>
      {hint && <p className="mt-0.5 text-[0.6875rem] text-tertiary">{hint}</p>}
    </div>
  );
}

function relativeTime(ms: number): string {
  const diff = Date.now() - ms;
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

function formatDurationShort(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s}s`;
}

function KillIcon() {
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
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}
