// Service favourites — persisted in localStorage, subscribed via
// useSyncExternalStore. Survives page reloads.

import { useSyncExternalStore } from "react";

const STORAGE_KEY = "ananke-favourite-services";

function load(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return new Set();
    return new Set(parseFavourites(JSON.parse(raw)));
  } catch {
    return new Set();
  }
}

/// Stored favourites, checked: anything but an array of strings is
/// treated as none rather than trusted.
export function parseFavourites(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is string => typeof v === "string");
}

function save(favs: Set<string>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...favs]));
  } catch {
    // localStorage unavailable.
  }
}

let favourites: Set<string> = load();
const listeners = new Set<() => void>();

function emit(): void {
  for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

function getSnapshot(): Set<string> {
  return favourites;
}

export function toggleFavourite(name: string): void {
  const next = new Set(favourites);
  if (next.has(name)) next.delete(name);
  else next.add(name);
  favourites = next;
  save(next);
  emit();
}

export function useFavourites(): Set<string> {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
