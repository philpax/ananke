// Bottom tab bar for mobile navigation. Icon-only, fixed to the
// bottom of the screen.

import { NavLink, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { ChatIcon, DashboardIcon, StatsIcon } from "../ui/icons.tsx";

const MOBILE_NAV_ITEMS = [
  {
    to: "/",
    labelKey: "nav.dashboard",
    icon: <DashboardIcon className="w-5 h-5" />,
  },
  { to: "/chat", labelKey: "nav.chat", icon: <ChatIcon className="w-5 h-5" /> },
  {
    to: "/stats",
    labelKey: "nav.stats",
    icon: <StatsIcon className="w-5 h-5" />,
  },
];

export function MobileNav() {
  const { t } = useTranslation();
  const location = useLocation();

  const serviceMatch = location.pathname.match(/^\/services\/(.+)$/);
  const chatTo = serviceMatch
    ? `/chat?model=${encodeURIComponent(serviceMatch[1])}`
    : "/chat";

  const navItems = MOBILE_NAV_ITEMS.map((item) =>
    item.to === "/chat" ? { ...item, to: chatTo } : item,
  );

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-10 flex h-16 border-t border-border-default bg-surface pb-[env(safe-area-inset-bottom)]">
      {navItems.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to === "/"}
          className={({ isActive }) =>
            `flex flex-1 flex-col items-center justify-center gap-0.5 text-xs transition-colors ${
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
