// The service list's pure rules: which services a search and the
// favourites filter keep, the order they sort in, and the state groups
// they fall into.

import type { ServiceStateKind, ServiceSummary } from "../../../api/client.ts";

export type SortOrder = "alpha" | "recent" | "size";

export const SORT_ORDERS: readonly SortOrder[] = ["alpha", "recent", "size"];

export function parseSortOrder(raw: string | null): SortOrder {
  return SORT_ORDERS.find((order) => order === raw) ?? "alpha";
}

export type ListFilter = {
  readonly query: string;
  readonly favouritesOnly: boolean;
  readonly favourites: ReadonlySet<string>;
};

/// Services whose name or description contains the query, ignoring case,
/// limited to favourites when asked.
export function filterServices(
  services: readonly ServiceSummary[],
  { query, favouritesOnly, favourites }: ListFilter,
): ServiceSummary[] {
  const needle = query.trim().toLowerCase();
  return services.filter(
    (s) =>
      (!favouritesOnly || favourites.has(s.name)) &&
      (needle === "" ||
        s.name.toLowerCase().includes(needle) ||
        (s.description?.toLowerCase().includes(needle) ?? false)),
  );
}

/// Favourites first, then by the chosen order, with the name as the
/// tiebreak so the order is stable. A service with no footprint sorts
/// last by size; one never used sorts last by recency.
export function sortServices(
  services: readonly ServiceSummary[],
  order: SortOrder,
  favourites: ReadonlySet<string>,
): ServiceSummary[] {
  return [...services].sort((a, b) => {
    const fav = Number(favourites.has(b.name)) - Number(favourites.has(a.name));
    if (fav !== 0) return fav;
    const byOrder = compareBy(order, a, b);
    return byOrder !== 0 ? byOrder : a.name.localeCompare(b.name);
  });
}

/// The order state groups appear in: what is busy, then what needs
/// attention, then what is quiet. A `Record` so that a new state is a
/// compile error here rather than an unsorted group.
export const STATE_GROUP_ORDER: Readonly<Record<ServiceStateKind, number>> = {
  running: 0,
  starting: 1,
  draining: 2,
  failed: 3,
  disabled: 4,
  idle: 5,
  stopped: 6,
  evicted: 7,
  unknown: 8,
};

export type StateGroup = {
  readonly state: ServiceStateKind;
  readonly services: readonly ServiceSummary[];
};

/// Already-sorted services split into state groups, in group order, each
/// keeping the incoming order. Empty groups are left out.
export function groupByState(sorted: readonly ServiceSummary[]): StateGroup[] {
  const groups = new Map<ServiceStateKind, ServiceSummary[]>();
  for (const service of sorted) {
    const group = groups.get(service.state);
    if (group) group.push(service);
    else groups.set(service.state, [service]);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => STATE_GROUP_ORDER[a] - STATE_GROUP_ORDER[b])
    .map(([state, services]) => ({ state, services }));
}

function compareBy(
  order: SortOrder,
  a: ServiceSummary,
  b: ServiceSummary,
): number {
  switch (order) {
    case "alpha":
      return 0;
    case "recent":
      return (b.last_used_ms ?? -1) - (a.last_used_ms ?? -1);
    case "size":
      return (b.footprint_bytes ?? -1) - (a.footprint_bytes ?? -1);
    default: {
      const unreachable: never = order;
      return unreachable;
    }
  }
}
