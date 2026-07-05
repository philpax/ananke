// Auto-scroll a container to the bottom when new content arrives, but only
// while the user is pinned to the bottom — scrolling up detaches so history
// stays readable, and scrolling back to the bottom re-attaches. Shared by the
// logs, events, and chat views, which previously each open-coded this (two
// correctly, one not).
//
// `dep` is the value that signals new content (e.g. line count, message
// array); the layout effect re-runs and, if still pinned, snaps to the bottom
// before paint (so a burst of new content never flickers scrolled-up first).
//
// Detach is keyed on scroll *direction*, not distance from the bottom: only a
// genuine upward scroll (scrollTop decreasing) unpins. Content growing below
// the viewport and our own programmatic snaps both only increase scrollTop, so
// they can't spuriously detach — which is the bug a distance-only check hits
// when a batch of lines lands and the async scroll event from the snap fires
// after the next batch has already grown the content.
//
// `pinned` is exposed as reactive state so callers can reflect the follow
// state in the UI; it only re-renders on the detach/attach transition (setting
// state to its current value is a no-op in React). `scrollToBottom` is an
// imperative re-pin for a "jump to latest" affordance.

import { useLayoutEffect, useRef, useState } from "react";

// Pixels from the bottom within which the view still counts as pinned.
const NEAR_BOTTOM_PX = 32;
// Ignore sub-pixel scrollTop jitter when deciding if the user scrolled up.
const UP_TOLERANCE_PX = 1;

export function useStickToBottom(dep: unknown) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const lastTopRef = useRef(0);
  const [pinned, setPinned] = useState(true);

  function setPinnedState(next: boolean) {
    pinnedRef.current = next;
    setPinned(next);
  }

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
    const scrolledUp = el.scrollTop < lastTopRef.current - UP_TOLERANCE_PX;
    lastTopRef.current = el.scrollTop;
    // Reaching the bottom re-pins; an intentional upward scroll detaches.
    // Everything else (content growth, our own snap) leaves the pin as-is.
    if (atBottom) setPinnedState(true);
    else if (scrolledUp) setPinnedState(false);
  }

  function scrollToBottom() {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    lastTopRef.current = el.scrollTop;
    setPinnedState(true);
  }

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (pinnedRef.current && el) {
      el.scrollTop = el.scrollHeight;
      lastTopRef.current = el.scrollTop;
    }
  }, [dep]);

  return { scrollRef, onScroll, pinned, scrollToBottom };
}
