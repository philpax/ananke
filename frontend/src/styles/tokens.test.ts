// Holds the WCAG contrast targets for the design tokens in both themes.
// Parses `tokens.css` directly, resolving `var()`, `light-dark()`, and
// `hsl()` at the declared hue, so a changed value is checked where it
// is written.

import { describe, expect, it } from "vitest";

import tokensCss from "./tokens.css?raw";

type Theme = "light" | "dark";
type Rgb = readonly [number, number, number];

const THEMES: readonly Theme[] = ["light", "dark"];

const declarations = parseDeclarations(tokensCss);

describe("token contrast", () => {
  const cases: readonly (readonly [
    string,
    readonly string[],
    readonly string[],
    number,
  ])[] = [
    [
      "strong text",
      ["text", "text-body", "text-soft"],
      ["tone-canvas", "tone-surface", "tone-raised"],
      7,
    ],
    [
      "muted text",
      ["text-muted"],
      ["tone-canvas", "tone-surface", "tone-raised", "tone-selected-soft"],
      4.5,
    ],
    ["faint text", ["text-faint"], ["tone-surface"], 3],
    ["accent", ["accent"], ["tone-surface", "tone-canvas"], 4.5],
    ["text on selected", ["accent-text"], ["tone-selected"], 7],
    ["text on accent", ["text-on-accent"], ["accent-strong"], 4.5],
    ["ok", ["status-ok"], ["status-ok-bg", "tone-surface"], 4.5],
    ["warn", ["status-warn"], ["status-warn-bg", "tone-surface"], 4.5],
    ["danger", ["status-danger"], ["status-danger-bg", "tone-surface"], 4.5],
    ["modality", ["chip-vision", "chip-embedding"], ["tone-surface"], 4.5],
    [
      "series",
      ["series-1", "series-2", "series-3", "series-4", "series-5"],
      ["tone-surface"],
      3,
    ],
    ["focus ring", ["accent"], ["tone-surface", "tone-canvas"], 3],
    ["input boundary", ["line-control"], ["tone-canvas", "tone-surface"], 3],
  ];

  for (const theme of THEMES) {
    for (const [label, fgs, bgs, min] of cases) {
      for (const fg of fgs) {
        for (const bg of bgs) {
          it(`${theme}: ${label} ${fg} on ${bg} is at least ${min}:1`, () => {
            const ratio = contrast(
              resolveColor(fg, theme),
              resolveColor(bg, theme),
            );
            expect(ratio).toBeGreaterThanOrEqual(min);
          });
        }
      }
    }
  }
});

describe("token parsing", () => {
  it("picks the matching side of light-dark()", () => {
    expect(resolveColor("tone-canvas", "light")).toEqual([243, 243, 245]);
    expect(resolveColor("tone-canvas", "dark")).toEqual([16, 17, 20]);
  });
});

function parseDeclarations(css: string): ReadonlyMap<string, string> {
  const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const map = new Map<string, string>();
  for (const match of withoutComments.matchAll(/--([\w-]+)\s*:\s*([^;]+);/g)) {
    const [, name, value] = match;
    if (name !== undefined && value !== undefined) {
      map.set(name, value.replace(/\s+/g, " ").trim());
    }
  }
  return map;
}

function resolveColor(name: string, theme: Theme): Rgb {
  return parseColor(resolveValue(name, theme, 0));
}

function resolveValue(name: string, theme: Theme, depth: number): string {
  const raw = declarations.get(name);
  if (raw === undefined) throw new Error(`unknown token --${name}`);
  if (depth > 10) throw new Error(`token --${name} is circular`);
  const substituted = raw.replace(/var\(--([\w-]+)\)/g, (_, inner: string) =>
    resolveValue(inner, theme, depth + 1),
  );
  return evaluateCalc(pickLightDark(substituted, theme));
}

function pickLightDark(value: string, theme: Theme): string {
  const start = value.indexOf("light-dark(");
  if (start < 0) return value;
  const open = start + "light-dark(".length;
  let depth = 0;
  let comma = -1;
  let end = -1;
  for (let i = open; i < value.length && end < 0; i++) {
    const ch = value[i];
    if (ch === "(") depth++;
    else if (ch === ")" && depth === 0) end = i;
    else if (ch === ")") depth--;
    else if (ch === "," && depth === 0) comma = i;
  }
  if (comma < 0 || end < 0) throw new Error(`malformed light-dark: ${value}`);
  const branch =
    theme === "light" ? value.slice(open, comma) : value.slice(comma + 1, end);
  return pickLightDark(
    value.slice(0, start) + branch.trim() + value.slice(end + 1),
    theme,
  );
}

function evaluateCalc(value: string): string {
  return value.replace(
    /calc\(\s*(-?[\d.]+)\s*([+-])\s*(-?[\d.]+)\s*\)/g,
    (_, a: string, op: string, b: string) =>
      String(op === "+" ? Number(a) + Number(b) : Number(a) - Number(b)),
  );
}

function parseColor(value: string): Rgb {
  const hex = /^#([0-9a-f]{6})$/i.exec(value);
  if (hex?.[1] !== undefined) {
    const n = parseInt(hex[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const hsl = /^hsl\(\s*([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*\)$/.exec(value);
  if (hsl?.[1] !== undefined && hsl[2] !== undefined && hsl[3] !== undefined) {
    return hslToRgb(Number(hsl[1]), Number(hsl[2]) / 100, Number(hsl[3]) / 100);
  }
  throw new Error(`cannot parse colour ${value}`);
}

function hslToRgb(h: number, s: number, l: number): Rgb {
  const a = s * Math.min(l, 1 - l);
  const channel = (n: number) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [channel(0), channel(8), channel(4)];
}

function contrast(a: Rgb, b: Rgb): number {
  const la = luminance(a);
  const lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

function luminance([r, g, b]: Rgb): number {
  const lin = (c: number) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}
