// Services: the list on the left, the selected service on the right. One
// element serves both `/` and `/services/:name`, so the list keeps its
// scroll and search while services are opened.
//
// Below `lg` the URL alone decides which pane shows: the list at `/`, the
// detail at a service, so Back works with no state to resync. From `lg` up
// both show, the list resizable and hideable. The nav rail is a fixed width
// from `md`, so the window breakpoint is a faithful stand-in for the room
// the panes have; inside them, layout follows container queries.

import { useEffect, useRef, type CSSProperties } from "react";
import { useTranslation } from "react-i18next";
import { useLocation } from "react-router-dom";

import { useServices } from "../../api/hooks.ts";
import { cn } from "../ui/cn.ts";
import { serviceFromPath } from "../layout/nav/navigation.ts";
import { SelectServiceEmpty } from "./SelectServiceEmpty.tsx";
import { ServiceDetail } from "./detail/ServiceDetail.tsx";
import { ServiceList } from "./list/ServiceList.tsx";
import { ListSeparator } from "./pane/ListSeparator.tsx";
import { useListPane } from "./pane/useListPane.ts";

export function ServicesView() {
  const { t } = useTranslation();
  const { pathname } = useLocation();
  const selected = serviceFromPath(pathname);
  const services = useServices();
  const pane = useListPane();
  const count = services.data?.length ?? 0;
  const browse = () => pane.setHidden(false);
  // Hiding the list removes the control that hid it, so focus moves to the
  // Browse button that brings the list back instead of falling to the body.
  const focusBrowse = useRef(false);
  const hide = () => {
    focusBrowse.current = true;
    pane.setHidden(true);
  };
  useEffect(() => {
    if (!pane.hidden || !focusBrowse.current) return;
    focusBrowse.current = false;
    document.querySelector<HTMLElement>(`[${BROWSE_ATTR}]`)?.focus();
  }, [pane.hidden]);
  // The stored width reaches CSS as a property, so the 50% cap and the
  // bounds stay in the stylesheet and nothing measures the container.
  const listStyle: CSSProperties & Record<"--list-w", string> = {
    "--list-w": `${pane.width}px`,
  };

  return (
    <div className="flex h-full min-h-0">
      <section
        id={LIST_ID}
        aria-label={t("services.title")}
        style={listStyle}
        className={cn(
          "@container/list relative flex min-h-0 flex-col border-r border-line-strong bg-surface max-lg:w-full lg:w-[clamp(15rem,var(--list-w),min(37.5rem,50%))] lg:shrink-0",
          selected !== undefined && "max-lg:hidden",
          pane.hidden && "lg:hidden",
        )}
      >
        <ServiceList selected={selected} onHide={hide} />
        {!pane.hidden && (
          <ListSeparator
            width={pane.width}
            controls={LIST_ID}
            onResize={pane.setWidth}
            onHide={hide}
          />
        )}
      </section>
      <div
        className={cn(
          "min-h-0 min-w-0 flex-1",
          selected === undefined && "max-lg:hidden",
        )}
      >
        {selected !== undefined ? (
          <ServiceDetail
            key={selected}
            name={selected}
            summary={services.data?.find((s) => s.name === selected)}
            listHidden={pane.hidden}
            serviceCount={count}
            onBrowse={browse}
          />
        ) : (
          <SelectServiceEmpty
            listHidden={pane.hidden}
            count={count}
            onBrowse={browse}
          />
        )}
      </div>
    </div>
  );
}

const LIST_ID = "services-list";
const BROWSE_ATTR = "data-browse-services";
