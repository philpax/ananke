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
    to: "/services",
    labelKey: "nav.services",
    icon: <ServicesIcon />,
  },
  {
    to: "/devices",
    labelKey: "nav.devices",
    icon: <DevicesIcon />,
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
    to: "/metrics",
    labelKey: "nav.metrics",
    icon: <MetricsIcon />,
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
    <aside className="hidden md:flex w-14 lg:w-48 flex-col border-r border-border-default bg-surface">
      <div className="flex items-center gap-2 px-3 py-3 border-b border-border-default">
        <span className="text-sm font-bold text-primary">ananke</span>
      </div>

      <nav className="flex-1 py-2">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 text-sm transition-colors ${
                isActive
                  ? "text-primary bg-elevated"
                  : "text-secondary hover:text-primary hover:bg-elevated"
              }`
            }
          >
            <span className="shrink-0">{item.icon}</span>
            <span className="hidden lg:inline">{t(item.labelKey)}</span>
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border-default px-3 py-2">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${healthVariant}`} />
          <span className="hidden lg:inline text-xs text-tertiary font-mono">
            {window.location.host}
          </span>
        </div>
      </div>
    </aside>
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

function ServicesIcon({ className = baseClass }: IconProps) {
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
      <path d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  );
}

function DevicesIcon({ className = baseClass }: IconProps) {
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
      <rect width="20" height="14" x="2" y="3" rx="2" />
      <path d="M8 21h8M12 17v4" />
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

function MetricsIcon({ className = baseClass }: IconProps) {
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
