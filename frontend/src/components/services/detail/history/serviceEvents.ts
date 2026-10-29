// A service's recent events: the live event history merged with the
// persisted restart log, which reaches back further than the daemon's
// in-memory window.

import type {
  DaemonEvent,
  ServiceRestartEntry,
} from "../../../../api/client.ts";
import { mergeRestarts } from "../../../events/eventLog.ts";

/// Events about `name` within `[since, end]`, newest first. Allocation
/// changes are dropped: a dynamic service emits one per balloon step, which
/// would bury everything else. A persisted restart that the live history
/// also holds appears once.
export function serviceEvents(
  history: readonly DaemonEvent[],
  restarts: readonly ServiceRestartEntry[],
  name: string,
  since: number,
  end: number,
): DaemonEvent[] {
  const inRange = (at: number) => at >= since && at <= end;
  return mergeRestarts(
    history.filter((e) => concernsService(e, name)),
    restarts.filter((r) => r.service === name),
  )
    .filter((e) => inRange(e.at_ms))
    .sort((a, b) => b.at_ms - a.at_ms);
}

function concernsService(event: DaemonEvent, name: string): boolean {
  switch (event.type) {
    case "state_changed":
    case "estimator_drift":
    case "auto_restarted":
      return event.service === name;
    case "config_reloaded":
      return event.changed_services.includes(name);
    case "reservation_placed":
      return event.evicted.includes(name);
    case "allocation_changed":
    case "reservation_removed":
      return false;
    default: {
      const unreachable: never = event;
      return unreachable;
    }
  }
}
