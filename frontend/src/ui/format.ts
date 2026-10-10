export function memory(bytes: number | null | undefined, digits = 2) {
  if (bytes == null) return "—";
  return `${(bytes / 1024 ** 3).toFixed(digits)} GiB`;
}

export function number(value: number | null | undefined, digits = 1) {
  return value == null || !Number.isFinite(value)
    ? "—"
    : value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function time(ms: number) {
  return new Date(ms).toLocaleTimeString(undefined, { hour12: false });
}

export function uptime(ms: number) {
  const hours = Math.floor(ms / 3_600_000);
  if (hours >= 24) return `${Math.floor(hours / 24)}d ${hours % 24}h`;
  if (hours) return `${hours}h ${Math.floor(ms / 60_000) % 60}m`;
  return `${Math.floor(ms / 60_000)}m`;
}

export function stateLabel(state: string) {
  if (state.startsWith("disabled_"))
    return `Disabled · ${state.slice(9).replaceAll("_", " ")}`;
  return state.charAt(0).toUpperCase() + state.slice(1).replaceAll("_", " ");
}

export function stateTone(state: string) {
  if (state === "running") return "ok";
  if (state === "failed" || state.startsWith("disabled")) return "danger";
  if (["starting", "draining", "backoff"].includes(state)) return "warn";
  return "neutral";
}
