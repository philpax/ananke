// Persistent sidebar for desktop navigation. Collapsible to icon-only.
// Shows the daemon endpoint and a health dot at the bottom.

import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { useServices } from "../../api/hooks.ts";

type NavItem = {
  to: string;
  labelKey: string;
  icon: ReactNode;
};

const NAV_ITEMS: NavItem[] = [
  {
    to: "/",
    labelKey: "nav.dashboard",
    icon: <DashboardIcon />,
  },
  {
    to: "/chat",
    labelKey: "nav.chat",
    icon: <ChatIcon />,
  },
  {
    to: "/oneshots",
    labelKey: "nav.oneshots",
    icon: <OneshotsIcon />,
  },
  {
    to: "/config",
    labelKey: "nav.config",
    icon: <ConfigIcon />,
  },
  {
    to: "/events",
    labelKey: "nav.events",
    icon: <EventsIcon />,
  },
  {
    to: "/stats",
    labelKey: "nav.stats",
    icon: <StatsIcon />,
  },
];

export function Sidebar() {
  const { t } = useTranslation();
  const services = useServices();

  const hasFailed = services.data?.some((s) => s.state === "failed");
  const hasRunning = services.data?.some((s) => s.state === "running");
  const healthVariant = hasFailed
    ? "bg-danger"
    : hasRunning
      ? "bg-success"
      : "bg-warning";

  return (
    <aside className="hidden md:flex w-14 lg:w-52 flex-col border-r border-border-default bg-surface">
      <div className="flex h-14 items-center gap-2.5 border-b border-border-default px-3 lg:px-4">
        <SpindleMark />
        <span className="hidden font-mono text-sm font-semibold tracking-[0.04em] text-primary lg:block">
          ananke
        </span>
      </div>

      <nav className="flex-1 py-3">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-2 text-sm transition-colors ${
                isActive
                  ? "bg-elevated/60 text-primary shadow-[inset_2px_0_0_var(--color-accent)]"
                  : "text-secondary hover:bg-elevated/40 hover:text-primary"
              }`
            }
          >
            <span className="shrink-0">{item.icon}</span>
            <span className="hidden lg:inline">{t(item.labelKey)}</span>
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border-default px-4 py-3">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${healthVariant}`} />
          <span className="hidden truncate font-mono text-xs text-tertiary lg:inline">
            {window.location.host}
          </span>
        </div>
      </div>
    </aside>
  );
}

// The spindle of Necessity — Ananke's whorl, rendered as an instrument
// dial. The lone brass mark in the chrome.
function SpindleMark() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      className="shrink-0 text-brass"
      aria-hidden="true"
    >
      <circle
        cx="12"
        cy="12"
        r="8"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <circle cx="12" cy="12" r="2.25" fill="currentColor" />
      <path
        d="M12 1.5v4M12 18.5v4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* --- Icons --- */

type IconProps = { className?: string };
const baseClass = "w-4 h-4";

function DashboardIcon({ className = baseClass }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="7" height="9" x="3" y="3" rx="1" />
      <rect width="7" height="5" x="14" y="3" rx="1" />
      <rect width="7" height="9" x="14" y="12" rx="1" />
      <rect width="7" height="5" x="3" y="16" rx="1" />
    </svg>
  );
}

function ChatIcon({ className = baseClass }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function OneshotsIcon({ className = baseClass }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 6v6l4 2" />
    </svg>
  );
}

function ConfigIcon({ className = baseClass }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function EventsIcon({ className = baseClass }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 12h4l3 9 4-18 3 9h4" />
    </svg>
  );
}

function StatsIcon({ className = baseClass }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 3v18h18" />
      <path d="M7 14l4-4 4 4 5-5" />
    </svg>
  );
}
