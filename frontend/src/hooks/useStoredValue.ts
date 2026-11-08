// A per-viewer preference in localStorage, shared by every component that
// reads the same key and kept in step with other tabs. Storage can be
// missing or throw (private windows, blocked site data), so every access
// is guarded and the parser supplies the fallback.

import { useSyncExternalStore } from "react";

export function useStoredValue<T>(
  key: string,
  parse: (raw: string | null) => T,
): readonly [T, (value: string) => void] {
  // The snapshot is the raw string, which is stable between renders;
  // parsing it here keeps the store free of per-type state.
  const raw = useSyncExternalStore(
    subscribe,
    () => read(key),
    () => null,
  );
  return [parse(raw), (value: string) => write(key, value)];
}

const listeners = new Set<() => void>();
// Writes are mirrored here, so a viewer without working storage still
// sees their choice for the life of the page.
const memory = new Map<string, string>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  window.addEventListener("storage", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

function read(key: string): string | null {
  try {
    return localStorage.getItem(key) ?? memory.get(key) ?? null;
  } catch {
    return memory.get(key) ?? null;
  }
}

function write(key: string, value: string): void {
  memory.set(key, value);
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage unavailable; the in-memory copy carries the value.
  }
  for (const listener of listeners) listener();
}
