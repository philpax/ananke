// The config editor's draft against the server's copy, as pure transitions.
// The draft is never replaced behind the operator's back: a newer version
// arriving while the draft has edits is held as a disk change for them to
// take or set aside, whether a reload event fetched it or a save found it.

import type { ConfigResponse } from "../../api/client.ts";

export type DiskChange = {
  /// The hash on disk now, which a save must name to overwrite it.
  readonly hash: string;
  /// The version on disk, once fetched. A rejected save knows only its
  /// hash until the refetch it triggers arrives.
  readonly server: ConfigResponse | null;
  /// `conflict` when a save was rejected for it, `reload` when it arrived
  /// on its own.
  readonly cause: "conflict" | "reload";
};

export type DraftState = {
  /// The server version the draft started from; a save names its hash.
  readonly base: ConfigResponse | null;
  /// When `base` was last read, in unix milliseconds.
  readonly readAtMs: number | null;
  readonly draft: string;
  readonly diskChange: DiskChange | null;
  /// The last save succeeded and nothing has been typed since.
  readonly saved: boolean;
};

export const INITIAL_DRAFT: DraftState = {
  base: null,
  readAtMs: null,
  draft: "",
  diskChange: null,
  saved: false,
};

export function isDirty(state: DraftState): boolean {
  return state.base !== null && state.draft !== state.base.content;
}

/// A fetch of the server's copy. Idempotent, so it can be applied on every
/// render as well as committed once.
export function receive(
  state: DraftState,
  server: ConfigResponse,
  readAtMs: number,
): DraftState {
  const { base } = state;
  if (base === null) return adopt(server, readAtMs);
  if (server.hash === base.hash || server.content === base.content) {
    // The same text, perhaps rehashed after a save: nothing to tell.
    return { ...state, base: server, readAtMs };
  }
  if (!isDirty(state) || server.content === state.draft) {
    return {
      ...adopt(server, readAtMs),
      saved: state.saved && server.content === state.draft,
    };
  }
  return {
    ...state,
    readAtMs,
    diskChange: {
      hash: server.hash,
      server,
      cause: state.diskChange?.cause ?? "reload",
    },
  };
}

export function edit(state: DraftState, draft: string): DraftState {
  return { ...state, draft, saved: false };
}

/// A save went through: the draft is what is on disk now. The new hash
/// arrives with the refetch the save triggers.
export function saved(state: DraftState, content: string): DraftState {
  if (state.base === null) return state;
  return { ...state, base: { ...state.base, content }, saved: true };
}

/// A save was rejected because the file changed underneath it.
export function conflict(state: DraftState, serverHash: string): DraftState {
  return {
    ...state,
    diskChange: {
      hash: serverHash,
      server:
        state.diskChange?.hash === serverHash ? state.diskChange.server : null,
      cause: "conflict",
    },
  };
}

/// Drop the draft for the newest known version on disk.
export function discard(state: DraftState): DraftState {
  const next = state.diskChange?.server ?? state.base;
  if (next === null) return state;
  return {
    ...state,
    base: next,
    draft: next.content,
    diskChange: null,
    saved: false,
  };
}

/// Keep the draft over a disk change: it now builds on the version on
/// disk, so the next save overwrites that version rather than failing.
export function keepDraft(state: DraftState): DraftState {
  const { base, diskChange } = state;
  if (base === null || diskChange === null) return state;
  return {
    ...state,
    base: diskChange.server ?? { ...base, hash: diskChange.hash },
    diskChange: null,
  };
}

function adopt(server: ConfigResponse, readAtMs: number): DraftState {
  return {
    base: server,
    readAtMs,
    draft: server.content,
    diskChange: null,
    saved: false,
  };
}
