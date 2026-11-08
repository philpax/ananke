// The section navigation: a rail with icons from `md` up, a scrolling
// tab row below it. One element for both, so every section is reachable
// at every width. Services carries a count of the services that failed or
// that the daemon disabled after a fault.

import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useLocation } from "react-router-dom";

import { useServices } from "../../../api/hooks.ts";
import { cn } from "../../ui/cn.ts";
import { Icon } from "../../ui/icon/Icon.tsx";
import type { IconName } from "../../ui/icon/iconGeometry.tsx";
import { rangeSearch } from "../../ui/form/timeRange.ts";
import { Chip } from "../../ui/status/Chip.tsx";
import { needsAttention } from "../../ui/status/stateTone.ts";
import { navItemVariants } from "./navItemVariants.ts";
import {
  SECTIONS,
  sectionForPath,
  sectionHref,
  serviceFromPath,
  type Section,
} from "./navigation.ts";

export function Nav({ className }: { readonly className?: string }) {
  const { t } = useTranslation();
  const { pathname, search } = useLocation();
  const range = rangeSearch(search);
  const services = useServices();
  const current = sectionForPath(pathname);

  // The last service opened, remembered across sections for this tab
  // only; the URL carries it across a reload.
  const viewing = serviceFromPath(pathname);
  const [lastName, setLastName] = useState(viewing);
  if (viewing !== undefined && viewing !== lastName) setLastName(viewing);
  const lastService = services.data?.find((s) => s.name === lastName);
  const attention = services.data?.filter(needsAttention).length;

  // Keep the active tab in view in the narrow tab row, so a deep link to
  // a section past the fold still shows which one is open.
  const activeRef = useRef<HTMLAnchorElement>(null);
  useEffect(() => {
    activeRef.current?.scrollIntoView?.({
      inline: "nearest",
      block: "nearest",
    });
  }, [pathname]);

  return (
    <nav
      aria-label={t("nav.label")}
      className={cn(
        "flex shrink-0 gap-0.5 bg-surface max-md:overflow-x-auto max-md:border-b max-md:border-line max-md:px-1.5 max-md:py-1 md:w-nav md:flex-col md:overflow-y-auto md:border-r-2 md:border-line-strong md:px-2 md:py-2.5",
        className,
      )}
    >
      {SECTIONS.map((section) => {
        const active = section === current;
        return (
          <Link
            key={section}
            ref={active ? activeRef : undefined}
            to={sectionHref(section, lastService, range, current)}
            aria-current={active ? "page" : undefined}
            className={navItemVariants({
              active,
              pinned: section === "config",
            })}
          >
            <Icon name={SECTION_ICON[section]} className="max-md:hidden" />
            {t(SECTION_LABEL[section])}
            {section === "services" &&
              attention !== undefined &&
              attention > 0 && (
                <Chip tone="danger" className="ml-auto font-normal">
                  <span aria-hidden="true">{attention}</span>
                  <span className="sr-only">
                    {t("nav.attentionServices", { count: attention })}
                  </span>
                </Chip>
              )}
          </Link>
        );
      })}
    </nav>
  );
}

const SECTION_ICON: Readonly<Record<Section, IconName>> = {
  services: "services",
  chat: "chat",
  events: "events",
  stats: "stats",
  config: "config",
};

const SECTION_LABEL: Readonly<Record<Section, string>> = {
  services: "nav.services",
  chat: "nav.chat",
  events: "nav.events",
  stats: "nav.stats",
  config: "nav.config",
};
