// Small formatting helpers shared across components.

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${n} B`;
}

// Render a parameter count the way HuggingFace and model card sites do:
// 7B, 32B, 235B for billions; M for millions; raw for anything smaller.
// One decimal when the value is below 100 of its unit so 7.2B reads
// naturally, no decimals above.
export function formatParameterCount(n: number): string {
  const fmt = (value: number, suffix: string) =>
    `${value < 100 ? value.toFixed(1) : Math.round(value)}${suffix}`;
  if (n >= 1e12) return fmt(n / 1e12, "T");
  if (n >= 1e9) return fmt(n / 1e9, "B");
  if (n >= 1e6) return fmt(n / 1e6, "M");
  if (n >= 1e3) return fmt(n / 1e3, "K");
  return `${n}`;
}

// Build a URL to a per-service proxy on whichever host the frontend
// is currently being served from. Matches the deployment where the
// daemon binds the frontend + management API on one host and exposes
// a per-service port for direct access.
export function serviceProxyUrl(port: number): string {
  return `${window.location.protocol}//${window.location.hostname}:${port}`;
}
