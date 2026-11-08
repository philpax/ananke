// The config draft held outside React, so unsaved edits survive leaving
// the Config view and coming back. One per tab; a reload of the page
// starts from the server's copy again.

import { useSyncExternalStore } from "react";

import { INITIAL_DRAFT, type DraftState } from "./configDraft.ts";

let state: DraftState = INITIAL_DRAFT;
const listeners = new Set<() => void>();

/// Apply one of the `configDraft` transitions.
export function updateDraft(transition: (s: DraftState) => DraftState): void {
  const next = transition(state);
  if (next === state) return;
  state = next;
  for (const listener of listeners) listener();
}

export function useDraft(): DraftState {
  return useSyncExternalStore(subscribe, getState, getState);
}

/// Back to a fresh tab's state, for tests.
export function resetDraft(): void {
  updateDraft(() => INITIAL_DRAFT);
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getState(): DraftState {
  return state;
}
