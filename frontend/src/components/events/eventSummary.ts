// One-line summaries of daemon events, shared by the events view and a
// service's recent events. Pure, so the wording is testable without a
// render.

import type { TFunction } from "i18next";

import type { DaemonEvent } from "../../api/client.ts";
import { formatBytes } from "../../util.ts";
import { reasonLabelKey, stateLabelKey } from "../ui/status/stateTone.ts";

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
    default: {
      const unreachable: never = event;
      return unreachable;
    }
  }
}
