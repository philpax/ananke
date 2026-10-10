import { useEffect, useState } from "preact/hooks";

import type { Range } from "../api/client";
import type { Device, Service } from "../api/contract";
import type { CapturedEvent } from "../api/system";
import type { Route } from "../layout/navigation";
import { Icon } from "../ui/Icon";
import { ServiceDetail } from "./ServiceDetail";
import { ServiceList } from "./ServiceList";
import { clampList, readFavourites } from "./list";

export function Services({
  active,
  services,
  devices,
  selected,
  width,
  range,
  onRange,
  onNavigate,
  captured,
  loading,
  error,
}: {
  readonly active: boolean;
  readonly services: readonly Service[];
  readonly devices: readonly Device[];
  readonly selected?: string;
  readonly width: number;
  readonly range: Range;
  readonly onRange: (r: Range) => void;
  readonly onNavigate: (r: Route) => void;
  readonly captured: readonly CapturedEvent[];
  readonly loading: boolean;
  readonly error?: string;
}) {
  const [listWidth, setListWidth] = useState(324),
    [listOpen, setListOpen] = useState(true),
    [showList, setShowList] = useState(!selected),
    [favourites, setFavourites] = useState(readFavourites);
  const multiple = width >= 1180;
  useEffect(() => {
    if (selected) setShowList(false);
  }, [selected, multiple]);
  useEffect(() => {
    try {
      localStorage.setItem("ananke.favourites.v12", JSON.stringify(favourites));
    } catch {
      /* Favourites remain usable when browser storage is unavailable. */
    }
  }, [favourites]);
  const actualWidth = clampList(listWidth, width);
  const browse = () => {
    setListOpen(true);
    setShowList(true);
  };
  const listVisible = multiple ? listOpen : showList || !selected;
  return (
    <div class="flex min-h-0 min-w-0 flex-1">
      <section
        aria-label="Services"
        class={
          listVisible
            ? `
              relative flex min-h-0 shrink-0 flex-col border-r
              border-line-strong bg-surface
            `
            : "hidden"
        }
        style={{ width: multiple ? `${actualWidth}px` : "100%" }}
      >
        <ServiceList
          services={services}
          selected={selected}
          favourites={favourites}
          onFavourite={(name) =>
            setFavourites((previous) =>
              previous.includes(name)
                ? previous.filter((n) => n !== name)
                : [...previous, name],
            )
          }
          onSelect={(service) => {
            setShowList(false);
            onNavigate({ section: "services", service });
          }}
          onHide={multiple ? () => setListOpen(false) : undefined}
          loading={loading}
          error={error}
        />
        {multiple && (
          <div
            role="separator"
            aria-label="Resize service list"
            aria-orientation="vertical"
            aria-valuemin={240}
            aria-valuemax={Math.min(600, width / 2)}
            aria-valuenow={Math.round(actualWidth)}
            tabIndex={0}
            title="Drag to resize service list"
            class="
              absolute inset-y-0 -right-[5px] z-10 w-2.5 cursor-col-resize
              touch-none
              after:absolute after:inset-y-0 after:left-1 after:w-px
              after:bg-line-strong
              hover:after:bg-accent
              focus-visible:after:bg-accent
            "
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                e.preventDefault();
                setListWidth(
                  clampList(
                    actualWidth + (e.key === "ArrowRight" ? 20 : -20),
                    width,
                  ),
                );
              }
            }}
            onPointerDown={(e) => {
              const target = e.currentTarget,
                start = e.clientX,
                initial = actualWidth;
              target.setPointerCapture(e.pointerId);
              const move = (event: PointerEvent) =>
                setListWidth(clampList(initial + event.clientX - start, width));
              const finish = () => {
                target.removeEventListener("pointermove", move);
                target.removeEventListener("pointerup", finish);
                target.removeEventListener("pointercancel", finish);
              };
              target.addEventListener("pointermove", move);
              target.addEventListener("pointerup", finish);
              target.addEventListener("pointercancel", finish);
            }}
          />
        )}
      </section>
      {active && selected && (multiple || !listVisible) ? (
        <ServiceDetail
          key={selected}
          name={selected}
          summary={services.find((s) => s.name === selected)}
          devices={devices}
          range={range}
          onRange={onRange}
          captured={captured}
          onNavigate={onNavigate}
          onBrowse={browse}
          singlePane={!multiple}
          listClosed={!listOpen}
          serviceCount={services.length}
        />
      ) : (
        multiple && (
          <main
            aria-label="No service selected"
            class="
              flex min-w-0 flex-1 items-center justify-center bg-canvas p-8
              text-muted
            "
          >
            <div
              class="
                flex max-w-[340px] flex-col items-center gap-2.5 text-center
                opacity-72
              "
            >
              <Icon name="empty" size={38} />
              <h2 class="mt-1 text-[15px] font-semibold text-soft">
                Select a service
              </h2>
              <p class="text-[13px] leading-[1.6]">
                Choose a service from the list to view its status, resource use,
                and activity.
              </p>
              {!listOpen && (
                <button
                  type="button"
                  class="text-[13px] font-semibold text-accent"
                  onClick={browse}
                >
                  Browse services
                </button>
              )}
            </div>
          </main>
        )
      )}
    </div>
  );
}
