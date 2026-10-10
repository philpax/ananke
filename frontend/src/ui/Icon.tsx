import type { ComponentChildren } from "preact";

export type IconName =
  | "services"
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
  | "disable"
  | "back"
  | "hide"
  | "browse"
  | "plus"
  | "edit"
  | "send"
  | "empty";

const geometry: Record<IconName, ComponentChildren> = {
  services: (
    <>
      <rect x="3" y="3" width="7" height="9" />
      <rect x="14" y="3" width="7" height="5" />
      <rect x="14" y="12" width="7" height="9" />
      <rect x="3" y="16" width="7" height="5" />
    </>
  ),
  chat: (
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  ),
  events: <path d="M3 12h4l3 9 4-18 3 9h4" />,
  stats: (
    <>
      <path d="M3 3v18h18" />
      <path d="M7 14l4-4 4 4 5-5" />
    </>
  ),
  config: (
    <>
      <circle cx="12" cy="12" r="3.4" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" />
    </>
  ),
  mark: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="2.3" fill="currentColor" stroke="none" />
      <path d="M12 1.5v4M12 18.5v4" stroke-linecap="round" />
    </>
  ),
  copy: (
    <>
      <rect x="9" y="9" width="12" height="12" />
      <path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" />
    </>
  ),
  check: <path d="M20 6L9 17l-5-5" />,
  star: (
    <path d="M12 3l2.9 5.9 6.1.9-4.5 4.3 1.1 6.1L12 17.3 6.4 20.2l1.1-6.1L3 9.8l6.1-.9z" />
  ),
  open: (
    <>
      <path d="M14 3h7v7" />
      <path d="M21 3l-9 9" />
      <path d="M19 14v6H4V5h6" />
    </>
  ),
  restart: (
    <>
      <path d="M21 12a9 9 0 1 1-3-6.7" />
      <path d="M21 4v5h-5" />
    </>
  ),
  stop: (
    <rect
      x="6"
      y="6"
      width="12"
      height="12"
      fill="currentColor"
      stroke="none"
    />
  ),
  start: <path d="M7 4l13 8-13 8z" fill="currentColor" stroke="none" />,
  disable: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="m6 6 12 12" />
    </>
  ),
  back: <path d="M15 5l-7 7 7 7" />,
  hide: (
    <>
      <path d="M15 5l-7 7 7 7" />
      <path d="M20 5v14" />
    </>
  ),
  browse: (
    <>
      <path d="M9 5l7 7-7 7" />
      <path d="M4 5v14" />
    </>
  ),
  plus: <path d="M12 5v14M5 12h14" />,
  edit: (
    <>
      <path d="M4 20h4l10-10-4-4L4 16z" />
      <path d="M14 6l4 4" />
    </>
  ),
  send: (
    <>
      <path d="M4 12h15" />
      <path d="M13 6l6 6-6 6" />
    </>
  ),
  empty: (
    <>
      <rect x="3" y="4" width="18" height="16" />
      <path d="M8 9h8M8 13h6M8 17h4" />
    </>
  ),
};

export function Icon({
  name,
  size = 16,
  filled = false,
}: {
  readonly name: IconName;
  readonly size?: number;
  readonly filled?: boolean;
}) {
  return (
    <svg
      aria-hidden="true"
      class="shrink-0"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      stroke-width={name === "empty" ? 1.25 : name === "mark" ? 1.6 : 1.8}
      stroke-linejoin={
        name === "copy" || name === "check" ? "round" : undefined
      }
    >
      {geometry[name]}
    </svg>
  );
}
