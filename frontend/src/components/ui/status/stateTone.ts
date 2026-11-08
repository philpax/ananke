// Service state to tone and translation key. A disabled service carries
// the reason it was disabled: an operator's choice is neutral, anything
// the daemon decided after a fault is danger.

import type { DisableReason, ServiceStateKind } from "../../../api/client.ts";
import type { Tone } from "../tone.ts";

export function stateTone(
  state: ServiceStateKind,
  reason?: DisableReason | null,
): Tone {
  switch (state) {
    case "running":
      return "ok";
    case "starting":
    case "draining":
      return "warn";
    case "failed":
      return "danger";
    case "disabled":
      return reason != null && isFaultReason(reason) ? "danger" : "neutral";
    case "idle":
    case "stopped":
    case "evicted":
    case "unknown":
      return "neutral";
    default: {
      const unreachable: never = state;
      return unreachable;
    }
  }
}

/// Whether the daemon disabled the service because something went wrong.
export function isFaultReason(reason: DisableReason): boolean {
  switch (reason) {
    case "user_disabled":
      return false;
    case "launch_failed":
    case "health_timeout":
    case "oom":
    case "auto_restart_loop":
      return true;
    default: {
      const unreachable: never = reason;
      return unreachable;
    }
  }
}

/// Whether a service needs an operator's attention: it failed, or the
/// daemon disabled it after a fault.
export function needsAttention(service: {
  readonly state: ServiceStateKind;
  readonly disabled_reason?: DisableReason | null;
}): boolean {
  return stateTone(service.state, service.disabled_reason) === "danger";
}

/// The `services.states.*` key for a state.
export function stateLabelKey(state: ServiceStateKind): string {
  return `services.states.${state}`;
}

/// The `services.disabledReasons.*` key for a disable reason.
export function reasonLabelKey(reason: DisableReason): string {
  return `services.disabledReasons.${reason}`;
}
