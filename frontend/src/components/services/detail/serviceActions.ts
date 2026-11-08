// What a service's header and row can do in its current state, where its
// Open link points, and how a lifecycle response reads to the operator.

import { isChatModel } from "../../../api/chatModels.ts";
import type {
  DisableResponse,
  EnableResponse,
  ServiceStateKind,
  ServiceSummary,
  StartResponse,
  StopResponse,
} from "../../../api/client.ts";

export type AvailableActions = {
  /// Start or Stop, whichever applies; `null` when neither does.
  readonly toggle: "start" | "stop" | null;
  readonly restart: boolean;
  readonly power: "enable" | "disable";
};

export function availableActions(state: ServiceStateKind): AvailableActions {
  switch (state) {
    case "starting":
    case "running":
    case "draining":
      return { toggle: "stop", restart: true, power: "disable" };
    case "idle":
    case "stopped":
    case "failed":
    case "evicted":
      return { toggle: "start", restart: false, power: "disable" };
    case "disabled":
      return { toggle: null, restart: false, power: "enable" };
    case "unknown":
      return { toggle: null, restart: false, power: "disable" };
    default: {
      const unreachable: never = state;
      return unreachable;
    }
  }
}

/// Whether Chat leads anywhere: the service is a chat model on the OpenAI
/// API and is not disabled, which would refuse the request.
export function canChat(
  service: Pick<ServiceSummary, "openai_compat" | "modality" | "state">,
): boolean {
  return isChatModel(service) && service.state !== "disabled";
}

/// The URL a service's public port answers on, from the host this tab
/// reached the daemon at. An IPv6 host is bracketed.
export function serviceOpenUrl(
  location: Pick<Location, "protocol" | "hostname">,
  port: number,
): string {
  const host =
    location.hostname.includes(":") && !location.hostname.startsWith("[")
      ? `[${location.hostname}]`
      : location.hostname;
  return `${location.protocol}//${host}:${port}/`;
}

export type ActionOutcome =
  | { readonly kind: "ok" }
  | { readonly kind: "error"; readonly message: string }
  | { readonly kind: "queue_full" };

/// A lifecycle response as success or a failure to show. `start` answers
/// 202 for both a queued start and two controlled failures, so a 2xx alone
/// does not mean it worked.
export function actionOutcome(
  response: StartResponse | StopResponse | EnableResponse | DisableResponse,
): ActionOutcome {
  if (!("status" in response)) return { kind: "ok" };
  switch (response.status) {
    case "unavailable":
      return { kind: "error", message: response.error.message };
    case "queue_full":
      return { kind: "queue_full" };
    default:
      return { kind: "ok" };
  }
}
