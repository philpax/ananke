// Bottom tab bar for mobile navigation. Icon-only, fixed to the
// bottom of the screen.

import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";

const MOBILE_NAV_ITEMS = [
  { to: "/", labelKey: "nav.dashboard", icon: <DashboardIcon /> },
  { to: "/chat", labelKey: "nav.chat", icon: <ChatIcon /> },
  { to: "/stats", labelKey: "nav.stats", icon: <StatsIcon /> },
];

export function MobileNav() {
  const { t } = useTranslation();

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-10 flex border-t border-border-default bg-surface">
      {MOBILE_NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to === "/"}
          className={({ isActive }) =>
            `flex flex-1 flex-col items-center gap-0.5 py-2 text-xs transition-colors ${
              isActive ? "text-accent" : "text-tertiary hover:text-secondary"
            }`
          }
        >
          <span className="shrink-0">{item.icon}</span>
          <span>{t(item.labelKey)}</span>
        </NavLink>
      ))}
    </nav>
  );
}

/* --- Icons (duplicated from Sidebar to keep MobileNav self-contained) --- */

type IconProps = { className?: string };
const baseClass = "w-5 h-5";

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
