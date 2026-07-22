// Persistent sidebar for desktop navigation. Collapsible to icon-only.
// Shows the daemon endpoint and a health dot at the bottom.

import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { useServices } from "../../api/hooks.ts";
import {
  ChatIcon,
  ConfigIcon,
  DashboardIcon,
  EventsIcon,
  OneshotsIcon,
  StatsIcon,
} from "../ui/icons.tsx";

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
  const location = useLocation();

  // When on a service detail page, carry the service name into the
  // Chat link so clicking Chat seeds the model from the service the
  // user was just viewing.
  const serviceMatch = location.pathname.match(/^\/services\/(.+)$/);
  const chatTo = serviceMatch
    ? `/chat?model=${encodeURIComponent(serviceMatch[1])}`
    : "/chat";

  const navItems = NAV_ITEMS.map((item) =>
    item.to === "/chat" ? { ...item, to: chatTo } : item,
  );

  const hasFailed = services.data?.some((s) => s.state === "failed");
  const hasRunning = services.data?.some((s) => s.state === "running");
  const healthVariant = hasFailed
    ? "bg-danger"
    : hasRunning
      ? "bg-success"
      : "bg-warning";

  return (
    <aside className="hidden md:flex w-14 lg:w-52 flex-col border-r border-border-default bg-surface">
      <div className="flex h-[var(--header-height)] items-center gap-2.5 border-b border-border-default px-3 lg:px-4">
        <SpindleMark />
        <span className="hidden font-mono text-sm font-semibold tracking-[0.04em] text-primary lg:block">
          ananke
        </span>
      </div>

      <nav className="flex-1 py-3">
        {navItems.map((item) => (
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
        <span className={`h-2 w-2 rounded-full ${healthVariant}`} />
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
