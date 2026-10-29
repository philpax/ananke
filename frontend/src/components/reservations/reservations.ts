// The reservations view's pure logic: how each device's memory splits
// between manual reservations and service pledges, and turning the form's
// text fields into a request. Kept apart from the view so both are testable
// without a render.

import type {
  CreateReservationRequest,
  DeviceSummary,
} from "../../api/client.ts";

const GIB = 1024 ** 3;

export type DeviceBudget = {
  readonly id: string;
  readonly name: string;
  readonly total: number;
  /// Held by manual reservations.
  readonly manual: number;
  /// Pledged to services ananke runs.
  readonly services: number;
  /// What the pledge book leaves for new placements.
  readonly unpledged: number;
};

export function deviceBudgets(
  devices: readonly DeviceSummary[],
): DeviceBudget[] {
  return devices.map((d) => {
    let manual = 0;
    let services = 0;
    for (const r of d.reservations) {
      if (r.reservation != null) manual += r.bytes;
      else services += r.bytes;
    }
    return {
      id: d.id,
      name: d.name,
      total: d.total_bytes,
      manual,
      services,
      unpledged: Math.max(0, d.total_bytes - manual - services),
    };
  });
}

export type ReservationForm = {
  readonly label: string;
  /// GiB per device id, as typed.
  readonly amounts: Readonly<Record<string, string>>;
};

export const EMPTY_FORM: ReservationForm = {
  label: "",
  amounts: {},
};

export type FormProblem =
  | { readonly kind: "no_devices" }
  | { readonly kind: "bad_amount"; readonly device: string }
  | { readonly kind: "too_large"; readonly device: string };

/// The request the form describes, or the first problem with it. Blank
/// amounts are skipped.
export function formRequest(
  form: ReservationForm,
  budgets: readonly DeviceBudget[],
):
  | { readonly ok: true; readonly request: CreateReservationRequest }
  | { readonly ok: false; readonly problem: FormProblem } {
  const devices: Record<string, number> = {};
  for (const budget of budgets) {
    const raw = form.amounts[budget.id]?.trim() ?? "";
    if (raw === "") continue;
    const gib = Number(raw);
    if (!Number.isFinite(gib) || gib <= 0) {
      return { ok: false, problem: { kind: "bad_amount", device: budget.id } };
    }
    const bytes = Math.ceil(gib * GIB);
    if (bytes > budget.total) {
      return { ok: false, problem: { kind: "too_large", device: budget.id } };
    }
    devices[budget.id] = bytes;
  }
  if (Object.keys(devices).length === 0) {
    return { ok: false, problem: { kind: "no_devices" } };
  }
  const label = form.label.trim();
  return {
    ok: true,
    request: {
      devices,
      ...(label === "" ? {} : { label }),
    },
  };
}

/// Device amounts in id order, for one line of text.
export function deviceEntries(
  devices: Readonly<Record<string, number>>,
): [string, number][] {
  return Object.entries(devices).sort(([a], [b]) => a.localeCompare(b));
}
