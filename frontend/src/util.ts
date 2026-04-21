// Small formatting helpers shared across components.

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${n} B`;
}

// Build a URL to a per-service proxy on whichever host the frontend
// is currently being served from. Matches the deployment where the
// daemon binds the frontend + management API on one host and exposes
// a per-service port for direct access.
export function serviceProxyUrl(port: number): string {
  return `${window.location.protocol}//${window.location.hostname}:${port}`;
}
