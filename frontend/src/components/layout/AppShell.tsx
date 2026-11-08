// The page grid every view renders into. One DOM tree at every width; the
// `md` breakpoint only moves things. Narrow: band, notice, tab row, then
// content, stacked. Wide: band and notice across the top, the rail beside
// the content below them.
//
// `main` gets a definite height and does not scroll itself in practice:
// each view owns its scroll region, so there is never a second scrollbar.

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { DaemonUpdatedNotice } from "./DaemonUpdatedNotice.tsx";
import { HardwareBand } from "./band/HardwareBand.tsx";
import { Nav } from "./nav/Nav.tsx";

export function AppShell({ children }: { readonly children: ReactNode }) {
  const { t } = useTranslation();
  return (
    <div className="grid h-dvh grid-rows-[auto_auto_auto_minmax(0,1fr)] bg-canvas pb-[env(safe-area-inset-bottom)] text-ink md:grid-cols-[var(--spacing-nav)_minmax(0,1fr)] md:grid-rows-[auto_auto_minmax(0,1fr)]">
      <a
        href="#main"
        className="sr-only z-10 border border-line-strong bg-raised px-3 py-2 text-sm text-ink focus:not-sr-only focus:absolute focus:top-2 focus:left-2"
      >
        {t("shell.skipToContent")}
      </a>
      <HardwareBand className="row-start-1 md:col-span-2" />
      <div className="row-start-2 md:col-span-2">
        <DaemonUpdatedNotice />
      </div>
      <Nav className="row-start-3 md:col-start-1" />
      <main
        id="main"
        tabIndex={-1}
        className="relative row-start-4 flex min-h-0 min-w-0 flex-col overflow-auto overscroll-contain md:col-start-2 md:row-start-3"
      >
        {children}
      </main>
    </div>
  );
}
