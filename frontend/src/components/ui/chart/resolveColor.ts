// Canvas drawing cannot take `var()`, and `getPropertyValue` returns a
// token's declared text (a `light-dark()` pair) rather than a colour. A
// hidden probe element inside the chart lets the browser resolve each
// token to a concrete `rgb()` for the current theme.

/// Resolves `--color-<name>` for each name; the returned lookup falls back
/// to `currentColor` for a name it was not given.
export function resolveTokenColors(
  host: HTMLElement,
  names: readonly string[],
): (name: string) => string {
  const probe = document.createElement("span");
  probe.style.display = "none";
  host.appendChild(probe);
  const resolved = new Map<string, string>();
  for (const name of names) {
    probe.style.color = `var(--color-${name})`;
    resolved.set(name, getComputedStyle(probe).color);
  }
  probe.remove();
  return (name) => resolved.get(name) ?? "currentColor";
}

/// The computed value of a custom property that is not a colour, such as
/// a font stack.
export function resolveTokenValue(host: HTMLElement, name: string): string {
  return getComputedStyle(host).getPropertyValue(name).trim();
}
