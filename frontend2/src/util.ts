// Formatting helpers shared across the dashboard. All user-facing
// formatting goes through here so units, precision, and rounding stay
// consistent.

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${n} B`;
}

export function formatBytesShort(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)}G`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)}M`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)}K`;
  return `${n}`;
}

export function formatParameterCount(n: number): string {
  const fmt = (value: number, suffix: string) =>
    `${value < 100 ? value.toFixed(1) : Math.round(value)}${suffix}`;
  if (n >= 1e12) return fmt(n / 1e12, "T");
  if (n >= 1e9) return fmt(n / 1e9, "B");
  if (n >= 1e6) return fmt(n / 1e6, "M");
  if (n >= 1e3) return fmt(n / 1e3, "K");
  return `${n}`;
}

export function formatDuration(ms: number): string {
  if (ms < 1) return "0ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s}s`;
}

export function formatTimestamp(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatTokenRate(tokensPerSecond: number): string {
  if (tokensPerSecond < 1) return tokensPerSecond.toFixed(2);
  if (tokensPerSecond < 100) return tokensPerSecond.toFixed(1);
  return Math.round(tokensPerSecond).toString();
}

export function serviceProxyUrl(port: number): string {
  return `${window.location.protocol}//${window.location.hostname}:${port}`;
}

// Construct the OpenAI API base URL from the daemon's `openai_listen`
// address (e.g. "0.0.0.0:7070"). Uses the browser's hostname (not
// 0.0.0.0) with the port from the listen address. CORS is enabled on
// the daemon, so cross-origin requests work.
export function openaiBaseUrlFromListen(listen: string | undefined): string {
  const parts = (listen ?? "0.0.0.0:7070").split(":");
  const port = parts[parts.length - 1];
  return `http://${window.location.hostname}:${port}`;
}
