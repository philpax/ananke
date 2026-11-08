// Icon shapes on a 24-unit grid, drawn with `currentColor`. Each entry
// carries its own stroke width and join, so `Icon` has no per-name
// special cases.

import type { ReactNode } from "react";

export type IconName =
  | "services"
  | "dashboard"
  | "chat"
  | "events"
  | "stats"
  | "config"
  | "mark"
  | "copy"
  | "check"
  | "star"
  | "open"
  | "restart"
  | "stop"
  | "start"
  | "power"
  | "disable"
  | "trash"
  | "back"
  | "hide"
  | "browse"
  | "plus"
  | "edit"
  | "send"
  | "chevron"
  | "attach"
  | "close"
  | "empty"
  | "sun"
  | "moon";

export type IconGeometry = {
  readonly shape: ReactNode;
  readonly strokeWidth: number;
  readonly strokeLinejoin?: "round";
  readonly strokeLinecap?: "round";
};

const STROKE = 1.8;

export const ICON_GEOMETRY: Readonly<Record<IconName, IconGeometry>> = {
  services: {
    strokeWidth: STROKE,
    shape: (
      <>
        <rect x="3" y="4" width="18" height="6" />
        <rect x="3" y="14" width="18" height="6" />
        <path d="M7 7h2M7 17h2" />
      </>
    ),
  },
  dashboard: {
    strokeWidth: STROKE,
    shape: (
      <>
        <rect x="3" y="3" width="7" height="9" />
        <rect x="14" y="3" width="7" height="5" />
        <rect x="14" y="12" width="7" height="9" />
        <rect x="3" y="16" width="7" height="5" />
      </>
    ),
  },
  chat: {
    strokeWidth: STROKE,
    shape: (
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    ),
  },
  events: { strokeWidth: STROKE, shape: <path d="M3 12h4l3 9 4-18 3 9h4" /> },
  stats: {
    strokeWidth: STROKE,
    shape: (
      <>
        <path d="M3 3v18h18" />
        <path d="M7 14l4-4 4 4 5-5" />
      </>
    ),
  },
  config: {
    strokeWidth: STROKE,
    shape: (
      <>
        <circle cx="12" cy="12" r="3.4" />
        <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" />
      </>
    ),
  },
  mark: {
    strokeWidth: 1.6,
    shape: (
      <>
        <circle cx="12" cy="12" r="8" />
        <circle cx="12" cy="12" r="2.3" fill="currentColor" stroke="none" />
        <path d="M12 1.5v4M12 18.5v4" strokeLinecap="round" />
      </>
    ),
  },
  copy: {
    strokeWidth: STROKE,
    strokeLinejoin: "round",
    shape: (
      <>
        <rect x="9" y="9" width="12" height="12" />
        <path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" />
      </>
    ),
  },
  check: {
    strokeWidth: STROKE,
    strokeLinejoin: "round",
    shape: <path d="M20 6L9 17l-5-5" />,
  },
  star: {
    strokeWidth: STROKE,
    shape: (
      <path d="M12 3l2.9 5.9 6.1.9-4.5 4.3 1.1 6.1L12 17.3 6.4 20.2l1.1-6.1L3 9.8l6.1-.9z" />
    ),
  },
  open: {
    strokeWidth: STROKE,
    shape: (
      <>
        <path d="M14 3h7v7" />
        <path d="M21 3l-9 9" />
        <path d="M19 14v6H4V5h6" />
      </>
    ),
  },
  restart: {
    strokeWidth: STROKE,
    shape: (
      <>
        <path d="M21 12a9 9 0 1 1-3-6.7" />
        <path d="M21 4v5h-5" />
      </>
    ),
  },
  stop: {
    strokeWidth: STROKE,
    shape: (
      <rect
        x="6"
        y="6"
        width="12"
        height="12"
        fill="currentColor"
        stroke="none"
      />
    ),
  },
  start: {
    strokeWidth: STROKE,
    shape: <path d="M7 4l13 8-13 8z" fill="currentColor" stroke="none" />,
  },
  power: {
    strokeWidth: STROKE,
    strokeLinecap: "round",
    shape: (
      <>
        <path d="M12 3v9" />
        <path d="M18.4 6.6a9 9 0 1 1-12.8 0" />
      </>
    ),
  },
  disable: {
    strokeWidth: STROKE,
    shape: (
      <>
        <circle cx="12" cy="12" r="8" />
        <path d="m6 6 12 12" />
      </>
    ),
  },
  trash: {
    strokeWidth: STROKE,
    shape: (
      <>
        <path d="M3 6h18" />
        <path d="M8 6V3h8v3" />
        <path d="M5 6l1 15h12l1-15" />
      </>
    ),
  },
  back: { strokeWidth: STROKE, shape: <path d="M15 5l-7 7 7 7" /> },
  hide: {
    strokeWidth: STROKE,
    shape: (
      <>
        <path d="M15 5l-7 7 7 7" />
        <path d="M20 5v14" />
      </>
    ),
  },
  browse: {
    strokeWidth: STROKE,
    shape: (
      <>
        <path d="M9 5l7 7-7 7" />
        <path d="M4 5v14" />
      </>
    ),
  },
  plus: { strokeWidth: STROKE, shape: <path d="M12 5v14M5 12h14" /> },
  chevron: { strokeWidth: STROKE, shape: <path d="M6 9l6 6 6-6" /> },
  close: { strokeWidth: STROKE, shape: <path d="M6 6l12 12M18 6L6 18" /> },
  attach: {
    strokeWidth: STROKE,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    shape: (
      <path d="M20 11.5l-8 8a5 5 0 01-7-7l8.5-8.5a3.5 3.5 0 015 5L10 17.5a2 2 0 01-3-3l7.5-7.5" />
    ),
  },
  edit: {
    strokeWidth: STROKE,
    shape: (
      <>
        <path d="M4 20h4l10-10-4-4L4 16z" />
        <path d="M14 6l4 4" />
      </>
    ),
  },
  send: {
    strokeWidth: STROKE,
    shape: (
      <>
        <path d="M4 12h15" />
        <path d="M13 6l6 6-6 6" />
      </>
    ),
  },
  empty: {
    strokeWidth: 1.25,
    shape: (
      <>
        <rect x="3" y="4" width="18" height="16" />
        <path d="M8 9h8M8 13h6M8 17h4" />
      </>
    ),
  },
  sun: {
    strokeWidth: STROKE,
    strokeLinecap: "round",
    shape: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
      </>
    ),
  },
  moon: {
    strokeWidth: STROKE,
    strokeLinejoin: "round",
    shape: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  },
};
