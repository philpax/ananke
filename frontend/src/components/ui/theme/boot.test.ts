// The inline boot script in `index.html` duplicates the storage key and
// the override parse from `theme.ts`, since it must run before any module
// loads. This runs it against seeded storage so the two cannot drift.

import { afterEach, describe, expect, it } from "vitest";

import indexHtml from "../../../../index.html?raw";
import { THEME_STORAGE_KEY } from "./theme.ts";

const script = /<script>([\s\S]*?)<\/script>/.exec(indexHtml)?.[1] ?? "";

function runBoot(): void {
  // The script is a fixed, checked-in file; this is how it runs in the
  // page too.
  new Function(script)();
}

afterEach(() => {
  localStorage.clear();
  delete document.documentElement.dataset.theme;
  document.documentElement.style.removeProperty("color-scheme");
});

describe("index.html boot script", () => {
  it("reads the same storage key as the store", () => {
    expect(script).toContain(JSON.stringify(THEME_STORAGE_KEY));
  });

  it.each(["light", "dark"] as const)("applies a stored %s override", (t) => {
    localStorage.setItem(THEME_STORAGE_KEY, t);
    runBoot();
    expect(document.documentElement.dataset.theme).toBe(t);
    expect(document.documentElement.style.colorScheme).toBe(t);
  });

  it("leaves the page following the OS otherwise", () => {
    localStorage.setItem(THEME_STORAGE_KEY, "system");
    runBoot();
    expect(document.documentElement.dataset.theme).toBeUndefined();
  });
});
