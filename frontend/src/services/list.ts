import type { Service } from "../api/contract";

export type Sort = "A–Z" | "Recent" | "Size";

export function serviceGroups(
  services: readonly Service[],
  query: string,
  favourites: readonly string[],
  favouritesOnly: boolean,
  sort: Sort,
) {
  const groups = new Map<string, Service[]>();
  const selected = services
    .filter(
      (s) =>
        s.name.toLowerCase().includes(query.toLowerCase()) &&
        (!favouritesOnly || favourites.includes(s.name)),
    )
    .sort((a, b) => {
      const favourite =
        Number(favourites.includes(b.name)) -
        Number(favourites.includes(a.name));
      if (favourite) return favourite;
      const rank =
        sort === "Recent"
          ? (b.last_used_ms ?? 0) - (a.last_used_ms ?? 0)
          : sort === "Size"
            ? (b.footprint_bytes ?? -1) - (a.footprint_bytes ?? -1)
            : 0;
      return rank || a.name.localeCompare(b.name);
    });
  for (const s of selected) {
    const group = s.state.startsWith("disabled") ? "disabled" : s.state;
    const rows = groups.get(group) ?? [];
    rows.push(s);
    groups.set(group, rows);
  }
  const order = [
    "running",
    "starting",
    "draining",
    "backoff",
    "failed",
    "idle",
    "disabled",
  ];
  return {
    shown: selected.length,
    groups: [...groups.entries()].sort(
      (a, b) => order.indexOf(a[0]) - order.indexOf(b[0]),
    ),
  };
}

export function readFavourites(): readonly string[] {
  try {
    const value: unknown = JSON.parse(
      localStorage.getItem("ananke.favourites.v12") ?? "[]",
    );
    return Array.isArray(value) &&
      value.every((item: unknown) => typeof item === "string")
      ? value
      : [];
  } catch {
    return [];
  }
}

export function clampList(width: number, viewport: number) {
  return Math.min(600, viewport / 2, Math.max(240, width));
}
