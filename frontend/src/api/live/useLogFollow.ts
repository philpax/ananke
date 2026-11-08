// React adapter for a `logs` follow. The follow starts once the caller has a
// position to start from (the seed page's `follow_after`) and runs until the
// service changes or the caller stops following; a later change of position,
// such as a seed refetch, does not restart it, since it already holds every
// line after the first one.

import { useEffect, useReducer, useRef } from "react";

import type { LogPosition } from "../client.ts";
import {
  EMPTY_FOLLOW,
  applyFollowMessage,
  resumeFeed,
  type FollowMessage,
  type FollowState,
} from "./logFollow.ts";
import { useLiveSocket } from "./useLive.ts";

type Action =
  | { readonly kind: "reset" }
  | { readonly kind: "resume" }
  | {
      readonly kind: "message";
      readonly message: FollowMessage;
      readonly cap: number;
    };

/// Follow `name`'s log after `after`. `undefined` means the position is not
/// known yet, so the follow waits; `null` follows from the oldest line.
export function useLogFollow(
  name: string | null,
  after: LogPosition | null | undefined,
  cap: number,
): FollowState {
  const socket = useLiveSocket();
  const [state, dispatch] = useReducer(reduce, EMPTY_FOLLOW);
  // The follow's own copy of what it has received, kept by the socket
  // callbacks alone so a reconnect resumes from the newest line even before
  // React renders it.
  const stateRef = useRef(EMPTY_FOLLOW);
  const afterRef = useRef(after);
  const capRef = useRef(cap);
  useEffect(() => {
    afterRef.current = after;
    capRef.current = cap;
  });

  const ready = after !== undefined;
  useEffect(() => {
    if (name === null || !ready) return;
    dispatch({ kind: "reset" });
    stateRef.current = EMPTY_FOLLOW;
    const feed = {
      kind: "logs" as const,
      service: name,
      after: afterRef.current ?? null,
    };
    const release = socket.follow({
      resume: () => {
        // A (re)sent follow backfills again before it is live.
        dispatch({ kind: "resume" });
        return resumeFeed(feed, stateRef.current);
      },
      onMessage: (message) => {
        stateRef.current = applyFollowMessage(
          stateRef.current,
          message,
          capRef.current,
        );
        dispatch({ kind: "message", message, cap: capRef.current });
      },
    });
    return () => {
      release();
      dispatch({ kind: "reset" });
    };
  }, [name, ready, socket]);

  return state;
}

function reduce(state: FollowState, action: Action): FollowState {
  switch (action.kind) {
    case "reset":
      return EMPTY_FOLLOW;
    case "resume":
      return state.caughtUp ? { ...state, caughtUp: false } : state;
    case "message":
      return applyFollowMessage(state, action.message, action.cap);
    default: {
      const unreachable: never = action;
      return unreachable;
    }
  }
}
