// The one mapping from a status tone to classes. Chips, state labels,
// button tones, group headers, and row rails all compose these rather
// than repeating the class strings.

import { cva } from "class-variance-authority";

export type Tone = "neutral" | "ok" | "warn" | "danger";

/// Bordered, filled box: state chips, status chips, lifecycle buttons.
export const toneBox = cva("border", {
  variants: {
    tone: {
      neutral: "border-line bg-surface text-body",
      ok: "border-ok-line bg-ok-bg text-ok",
      warn: "border-warn-line bg-warn-bg text-warn",
      danger: "border-danger-line bg-danger-bg text-danger",
    },
  },
});

/// Foreground only: dots, inline status text.
export const toneText = cva("", {
  variants: {
    tone: {
      neutral: "text-soft",
      ok: "text-ok",
      warn: "text-warn",
      danger: "text-danger",
    },
  },
});

/// A left rail marking a row's severity.
export const toneRail = cva("border-l-4", {
  variants: {
    tone: {
      neutral: "border-l-transparent",
      ok: "border-l-ok",
      warn: "border-l-warn",
      danger: "border-l-danger",
    },
  },
});
