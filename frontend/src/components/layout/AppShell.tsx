// Top-level layout shell. Sidebar on desktop, bottom tab bar on
// mobile, with the routed content filling the rest.

import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar.tsx";
import { MobileNav } from "./MobileNav.tsx";

type AppShellProps = {
  children: ReactNode;
};

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="flex h-screen flex-col bg-base text-primary">
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-auto pb-16 md:pb-0">{children}</main>
      </div>
      <MobileNav />
    </div>
  );
}
