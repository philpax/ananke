// One-line summaries of daemon events, shared by the events view and a
// service's recent events. Pure, so the wording is testable without a
// render.

import type { TFunction } from "i18next";

import type { DaemonEvent } from "../../api/client.ts";
import { formatBytes } from "../../util.ts";
import { reasonLabelKey, stateLabelKey } from "../ui/status/stateTone.ts";

/// A manual reservation's display name: its label, else its id.
export function reservationName(
  id: number,
  label: string | null | undefined,
  t: TFunction,
): string {
  return label ? label : t("reservations.unnamed", { id });
}

export function summarizeEvent(event: DaemonEvent, t: TFunction): string {
  switch (event.type) {
    case "state_changed": {
      const to = t(stateLabelKey(event.to));
      const reason = event.disabled_reason;
      return t("events.stateChanged", {
        from: t(stateLabelKey(event.from)),
        to:
          reason == null
            ? to
            : t("events.stateWithReason", {
                state: to,
                reason: t(reasonLabelKey(reason)),
              }),
      });
    }
    case "allocation_changed": {
      const held = Object.entries(event.reservations)
        .filter(([, bytes]) => bytes > 0)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([device, bytes]) =>
          t("events.reservedOn", { bytes: formatBytes(bytes), device }),
        );
      return held.length === 0
        ? t("events.allocationReleased")
        : t("events.allocationChanged", { devices: held.join(", ") });
    }
    case "config_reloaded":
      return event.changed_services.length === 0
        ? t("events.configReloaded")
        : t("events.configReloadedChanged", {
            count: event.changed_services.length,
            services: event.changed_services.join(", "),
          });
    case "estimator_drift":
      return t("events.estimatorDriftCorrection", {
        class: event.class,
        value: event.rolling_mean.toFixed(3),
      });
    case "auto_restarted":
      return event.detail
        ? t("events.autoRestarted", {
            trigger: event.trigger,
            detail: event.detail,
          })
        : t("events.autoRestartedShort", { trigger: event.trigger });
    case "reservation_placed": {
      const held = Object.entries(event.devices)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([device, bytes]) =>
          t("events.reservedOn", { bytes: formatBytes(bytes), device }),
        )
        .join(", ");
      const name = reservationName(event.id, event.label, t);
      return event.evicted.length === 0
        ? t("events.reservationPlaced", { name, devices: held })
        : t("events.reservationPlacedEvicting", {
            name,
            devices: held,
            services: event.evicted.join(", "),
          });
    }
    case "reservation_removed":
      return t("events.reservationRemoved", {
        name: reservationName(event.id, event.label, t),
      });
    default: {
      const unreachable: never = event;
      return unreachable;
    }
  }
}
