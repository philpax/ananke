// Reservations (`/reservations`): how each device's memory splits between
// manual reservations and service pledges, the manual reservations an
// operator placed, and the services holding memory. A manual reservation
// outranks every service, so placing one stops whatever is in the way; a
// service's row offers a stop as a shortcut, while its page stays the place
// to manage it.

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import {
  useCreateReservation,
  useDevices,
  useLifecycle,
  useRemoveReservation,
  useReservations,
} from "../../api/hooks.ts";
import type {
  ManualReservation,
  ServiceReservation,
} from "../../api/client.ts";
import { formatBytes, formatTimestamp } from "../../util.ts";
import { reservationName } from "../events/eventSummary.ts";
import { Button } from "../ui/button/Button.tsx";
import { MemoryBar } from "../ui/data/MemoryBar.tsx";
import { EmptyState } from "../ui/feedback/EmptyState.tsx";
import { Message } from "../ui/feedback/Message.tsx";
import { TextInput } from "../ui/form/TextInput.tsx";
import { PageHeader } from "../ui/layout/PageHeader.tsx";
import { Panel } from "../ui/layout/Panel.tsx";
import { Chip } from "../ui/status/Chip.tsx";
import { StateChip } from "../ui/status/StateChip.tsx";
import {
  EMPTY_FORM,
  deviceBudgets,
  deviceEntries,
  formRequest,
  type DeviceBudget,
  type FormProblem,
  type ReservationForm,
} from "./reservations.ts";

export function ReservationsView() {
  const { t } = useTranslation();
  const reservations = useReservations();
  const devices = useDevices();
  const [formOpen, setFormOpen] = useState(false);
  const [placed, setPlaced] = useState<string | null>(null);
  const budgets = deviceBudgets(devices.data ?? []);
  const manual = reservations.data?.manual ?? [];
  const services = reservations.data?.services ?? [];

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PageHeader
        title={t("reservations.title")}
        meta={`${t("reservations.metaManual", { count: manual.length })} · ${t(
          "reservations.metaServices",
          { count: services.length },
        )}`}
        actions={
          !formOpen && (
            <Button
              tone="primary"
              icon="plus"
              onClick={() => {
                setPlaced(null);
                setFormOpen(true);
              }}
            >
              {t("reservations.new")}
            </Button>
          )
        }
      />
      <div className="min-h-0 flex-1 overflow-auto bg-canvas">
        <div className="flex flex-col gap-4 p-4">
          {reservations.error && (
            <Message kind="error" text={reservations.error.message} />
          )}
          {devices.error && (
            <Message kind="error" text={devices.error.message} />
          )}
          <Panel heading={t("reservations.devices")}>
            {budgets.length === 0 ? (
              <Message kind="loading" />
            ) : (
              <ul className="flex flex-col gap-3">
                {budgets.map((b) => (
                  <DeviceRow key={b.id} budget={b} />
                ))}
              </ul>
            )}
          </Panel>
          {formOpen && (
            <NewReservation
              budgets={budgets}
              onClose={() => setFormOpen(false)}
              onPlaced={(text) => {
                setPlaced(text);
                setFormOpen(false);
              }}
            />
          )}
          {placed && (
            <p role="status" className="text-sm text-ok">
              {placed}
            </p>
          )}
          <Panel heading={t("reservations.manual")} padding="none">
            {reservations.isPending ? (
              <Message kind="loading" className="px-4" />
            ) : manual.length === 0 ? (
              <EmptyState icon="lock" message={t("reservations.noManual")} />
            ) : (
              <ul className="divide-y divide-line-subtle">
                {manual.map((r) => (
                  <ManualRow key={r.id} reservation={r} />
                ))}
              </ul>
            )}
          </Panel>
          <Panel heading={t("reservations.services")} padding="none">
            {reservations.isPending ? (
              <Message kind="loading" className="px-4" />
            ) : services.length === 0 ? (
              <EmptyState message={t("reservations.noServices")} />
            ) : (
              <ul className="divide-y divide-line-subtle">
                {services.map((r) => (
                  <ServiceRow key={r.service} reservation={r} />
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function DeviceRow({ budget }: { readonly budget: DeviceBudget }) {
  const { t } = useTranslation();
  const summary = [
    budget.manual > 0 &&
      t("reservations.reserved", { bytes: formatBytes(budget.manual) }),
    budget.services > 0 &&
      t("reservations.servicesBytes", { bytes: formatBytes(budget.services) }),
    t("reservations.freeOf", {
      free: formatBytes(budget.unpledged),
      total: formatBytes(budget.total),
    }),
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <li className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="font-mono text-sm font-semibold text-ink">
          {budget.id}
        </span>
        <span className="text-xs text-muted">{budget.name}</span>
      </div>
      <MemoryBar
        variant="placement"
        total={budget.total}
        others={budget.services}
        self={budget.manual}
        size="md"
        label={`${budget.id}: ${summary}`}
      />
      <span className="text-xs text-muted">{summary}</span>
    </li>
  );
}

function DeviceAmounts({
  devices,
}: {
  readonly devices: Readonly<Record<string, number>>;
}) {
  return (
    <span className="flex flex-wrap gap-1.5">
      {deviceEntries(devices).map(([device, bytes]) => (
        <Chip key={device} className="font-mono">
          {device} {formatBytes(bytes)}
        </Chip>
      ))}
    </span>
  );
}

function ManualRow({
  reservation,
}: {
  readonly reservation: ManualReservation;
}) {
  const { t } = useTranslation();
  const remove = useRemoveReservation();
  const name = reservationName(reservation.id, reservation.label, t);
  return (
    <li className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
      <div className="flex min-w-0 flex-col gap-1">
        <span className="text-sm font-semibold wrap-anywhere text-ink">
          {name}
        </span>
        <span className="text-xs text-muted">
          {t("reservations.manualMeta", {
            id: reservation.id,
            time: formatTimestamp(reservation.created_at_ms),
          })}
        </span>
      </div>
      <DeviceAmounts devices={reservation.devices} />
      <div className="ml-auto flex items-center gap-2">
        {remove.error && <Message kind="error" text={remove.error.message} />}
        <Button
          tone="stop"
          size="sm"
          icon="trash"
          disabled={remove.isPending}
          onClick={() => remove.mutate(reservation.id)}
          aria-label={t("reservations.removeNamed", { name })}
        >
          {t("reservations.remove")}
        </Button>
      </div>
    </li>
  );
}

function ServiceRow({
  reservation,
}: {
  readonly reservation: ServiceReservation;
}) {
  const { t } = useTranslation();
  const lifecycle = useLifecycle();
  const stoppable =
    reservation.state === "running" || reservation.state === "starting";
  return (
    <li className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
      <div className="flex min-w-0 flex-col gap-1">
        <Link
          to={`/services/${encodeURIComponent(reservation.service)}`}
          className="font-mono text-sm font-semibold wrap-anywhere text-ink hover:text-accent hover:underline"
        >
          {reservation.service}
        </Link>
        <span className="flex flex-wrap items-center gap-2 text-xs text-muted">
          <StateChip state={reservation.state} />
          {reservation.inflight_count > 0 &&
            t("reservations.inflight", { count: reservation.inflight_count })}
        </span>
      </div>
      <DeviceAmounts devices={reservation.devices} />
      <div className="ml-auto flex items-center gap-2">
        {lifecycle.error && (
          <Message kind="error" text={lifecycle.error.message} />
        )}
        {stoppable && (
          <Button
            tone="quiet"
            size="sm"
            icon="stop"
            disabled={lifecycle.isPending}
            onClick={() =>
              lifecycle.mutate({ action: "stop", name: reservation.service })
            }
            aria-label={t("reservations.stopNamed", {
              name: reservation.service,
            })}
          >
            {t("reservations.stop")}
          </Button>
        )}
      </div>
    </li>
  );
}

function NewReservation({
  budgets,
  onClose,
  onPlaced,
}: {
  readonly budgets: readonly DeviceBudget[];
  readonly onClose: () => void;
  readonly onPlaced: (text: string) => void;
}) {
  const { t } = useTranslation();
  const create = useCreateReservation();
  const [form, setForm] = useState<ReservationForm>(EMPTY_FORM);
  const [problem, setProblem] = useState<FormProblem | null>(null);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const parsed = formRequest(form, budgets);
    if (!parsed.ok) {
      setProblem(parsed.problem);
      return;
    }
    setProblem(null);
    // Answer straight away: the evicted services show as stopping in the
    // live list rather than holding the button for their shutdown.
    create.mutate(
      { ...parsed.request, wait: false },
      {
        onSuccess: ({ reservation, evicted }) => {
          const name = reservationName(reservation.id, reservation.label, t);
          onPlaced(
            evicted.length === 0
              ? t("reservations.placed", { name })
              : t("reservations.placedEvicting", {
                  name,
                  services: evicted.join(", "),
                }),
          );
        },
      },
    );
  };

  return (
    <Panel heading={t("reservations.newTitle")}>
      <form onSubmit={submit} className="flex flex-col gap-4">
        <label className="flex min-w-0 flex-col gap-1 text-sm text-body">
          {t("reservations.label")}
          <TextInput
            value={form.label}
            maxLength={200}
            placeholder={t("reservations.labelPlaceholder")}
            onChange={(e) => setForm({ ...form, label: e.target.value })}
          />
        </label>
        <fieldset className="flex flex-col gap-2">
          <legend className="mb-1 text-sm text-body">
            {t("reservations.amounts")}
          </legend>
          {budgets.map((b) => (
            <label
              key={b.id}
              className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm"
            >
              <span className="w-16 font-mono text-ink">{b.id}</span>
              <TextInput
                className="w-28"
                value={form.amounts[b.id] ?? ""}
                inputMode="decimal"
                mono
                placeholder="GiB"
                aria-label={t("reservations.amountFor", { device: b.id })}
                onChange={(e) =>
                  setForm({
                    ...form,
                    amounts: { ...form.amounts, [b.id]: e.target.value },
                  })
                }
              />
              <span className="text-xs text-muted">
                {t("reservations.amountHint", {
                  free: formatBytes(b.unpledged),
                })}
              </span>
            </label>
          ))}
        </fieldset>
        <p className="text-xs text-muted">{t("reservations.evictionNote")}</p>
        {problem && (
          <Message
            kind="error"
            text={t(`reservations.problem.${problem.kind}`, problem)}
          />
        )}
        {create.error && <Message kind="error" text={create.error.message} />}
        <div className="flex flex-wrap items-center gap-2">
          <Button type="submit" tone="primary" disabled={create.isPending}>
            {create.isPending
              ? t("reservations.placing")
              : t("reservations.place")}
          </Button>
          <Button tone="quiet" onClick={onClose} disabled={create.isPending}>
            {t("common.cancel")}
          </Button>
        </div>
      </form>
    </Panel>
  );
}
