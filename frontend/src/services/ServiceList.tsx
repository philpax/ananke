import clsx from "clsx";
import { useState } from "preact/hooks";

import type { Service } from "../api/contract";
import { Copy } from "../ui/Copy";
import { Icon } from "../ui/Icon";
import { Message } from "../ui/Message";
import { memory, stateLabel, stateTone } from "../ui/format";
import { serviceGroups } from "./list";
import type { Sort } from "./list";

export function ServiceList({
  services,
  selected,
  favourites,
  onFavourite,
  onSelect,
  onHide,
  loading,
  error,
}: {
  readonly services: readonly Service[];
  readonly selected?: string;
  readonly favourites: readonly string[];
  readonly onFavourite: (name: string) => void;
  readonly onSelect: (name: string) => void;
  readonly onHide?: () => void;
  readonly loading: boolean;
  readonly error?: string;
}) {
  const [query, setQuery] = useState(""),
    [sort, setSort] = useState<Sort>("A–Z"),
    [only, setOnly] = useState(false);
  const rows = serviceGroups(services, query, favourites, only, sort);
  return (
    <>
      <div class="flex flex-col gap-2.5 border-b border-line px-3.5 pt-3.5 pb-3">
        <div class="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1.5">
          <h1 class="m-0 text-base font-semibold">Services</h1>
          <span class="text-[12.5px] text-muted tabular-nums">
            {rows.shown} / {services.length} shown
          </span>
          {onHide && (
            <button
              type="button"
              title="Hide service list"
              aria-label="Hide service list"
              class="
                ml-auto flex size-[30px] items-center justify-center border-0
                bg-transparent text-muted
                hover:text-accent
              "
              onClick={onHide}
            >
              <Icon name="hide" size={15} />
            </button>
          )}
        </div>
        <input
          aria-label="Search services"
          placeholder="Search services"
          class="
            h-[34px] border border-line-strong bg-canvas px-2.5 font-mono
            text-[13px] text-ink
          "
          value={query}
          onInput={(e) => setQuery(e.currentTarget.value)}
        />
        <div class="flex flex-wrap items-center gap-2.5">
          <div role="group" aria-label="Sort services" class="flex gap-0.5">
            {(["A–Z", "Recent", "Size"] as const).map((s) => (
              <button
                key={s}
                type="button"
                aria-pressed={sort === s}
                class={clsx(
                  `h-[30px] border px-2.5 text-[12.5px]`,
                  sort === s
                    ? `border-accent bg-selected text-accent-text`
                    : `border-line-strong bg-surface text-body`,
                )}
                onClick={() => setSort(s)}
              >
                {s}
              </button>
            ))}
          </div>
          <label
            class="
              ml-auto flex cursor-pointer items-center gap-[7px] text-[12.5px]
              text-body
            "
          >
            <input
              type="checkbox"
              checked={only}
              onChange={(e) => setOnly(e.currentTarget.checked)}
              class="size-[15px] accent-accent-strong"
            />
            Favourites only
          </label>
        </div>
      </div>
      <div class="min-h-0 flex-1 overflow-auto">
        {error && (
          <div class="px-3.5">
            <Message error={error} />
          </div>
        )}
        {loading && !services.length && (
          <div class="px-3.5">
            <Message loading />
          </div>
        )}
        {rows.groups.map(([state, group]) => (
          <div key={state}>
            <h2
              class={clsx(
                `
                  sticky top-0 z-1 m-0 flex items-center gap-2 border-y
                  border-line bg-raised px-3.5 py-[9px] text-xs font-semibold
                  tracking-widest uppercase
                `,
                stateTone(state) === "ok"
                  ? `text-ok`
                  : stateTone(state) === "danger"
                    ? `text-danger`
                    : stateTone(state) === "warn"
                      ? `text-warn`
                      : `text-soft`,
              )}
            >
              <span
                aria-hidden="true"
                class="size-2 shrink-0 rounded-full bg-current"
              />
              {stateLabel(state)}
              <span class="font-normal tracking-normal text-muted">
                {group.length}
              </span>
            </h2>
            {group.map((service) => (
              <ServiceRow
                key={service.name}
                service={service}
                selected={selected === service.name}
                favourite={favourites.includes(service.name)}
                onFavourite={() => onFavourite(service.name)}
                onSelect={() => onSelect(service.name)}
              />
            ))}
          </div>
        ))}
        {!loading && !rows.shown && (
          <div class="px-3.5">
            <Message empty="No services match your filters." />
          </div>
        )}
      </div>
    </>
  );
}

function ServiceRow({
  service,
  selected,
  favourite,
  onFavourite,
  onSelect,
}: {
  readonly service: Service;
  readonly selected: boolean;
  readonly favourite: boolean;
  readonly onFavourite: () => void;
  readonly onSelect: () => void;
}) {
  const devices = [...(service.footprint_devices ?? [])].sort((a, b) =>
    a.device === "cpu"
      ? -1
      : b.device === "cpu"
        ? 1
        : a.device.localeCompare(b.device, undefined, { numeric: true }),
  );
  const vision =
    service.has_mmproj || service.ananke_metadata?.["vision"] === true;
  return (
    <div
      class={clsx(
        `
          grid grid-cols-[34px_minmax(0,1fr)] items-center gap-1.5 border-b
          border-l-4 border-b-line-subtle py-1.5 pr-3 pl-2
        `,
        selected
          ? `border-l-accent bg-selected-soft`
          : `border-l-transparent bg-surface`,
      )}
    >
      <button
        type="button"
        aria-pressed={favourite}
        aria-label={`${favourite ? "Remove" : "Add"} ${service.name} ${favourite ? "from" : "to"} favourites`}
        class={clsx(
          `
            flex size-[30px] items-center justify-center border-0 bg-transparent
            hover:text-warn
          `,
          favourite ? `text-warn` : `text-faint`,
        )}
        onClick={onFavourite}
      >
        <Icon name="star" size={14} filled={favourite} />
      </button>
      <div
        role="button"
        tabIndex={0}
        aria-current={selected ? "true" : undefined}
        class="
          flex min-w-0 cursor-pointer flex-col gap-1 px-1 py-[5px] text-left
        "
        onClick={onSelect}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onSelect();
          }
        }}
      >
        <span class="flex min-w-0 items-center gap-[5px]">
          <span
            title={service.name}
            class="min-w-0 font-mono text-[13px] leading-[1.35] wrap-anywhere"
          >
            {service.name}
          </span>
          <Copy text={service.name} />
        </span>
        <span
          class="
            flex min-w-0 flex-wrap items-center gap-1.5 text-[11.5px] text-muted
          "
        >
          <span class="text-[12.5px] font-semibold tabular-nums">
            {memory(service.footprint_bytes)}
          </span>
          {vision && (
            <span class="border border-[#416078] px-1 text-[#84c9f4]">
              vision
            </span>
          )}
          {service.modality === "embedding" && (
            <span class="border border-[#376b55] px-1 text-[#7ed6b0]">
              embedding
            </span>
          )}
        </span>
        <span
          class="
            flex min-w-0 flex-wrap items-center gap-x-[11px] gap-y-1 text-xs
            leading-[1.4] font-semibold tabular-nums
          "
        >
          {devices.map((d) => (
            <span
              key={d.device}
              class={clsx(
                `whitespace-nowrap`,
                d.device === "cpu"
                  ? `text-soft`
                  : d.device === "gpu:0"
                    ? `text-accent`
                    : `text-[#84c9f4]`,
              )}
            >
              {d.device}{" "}
              <b class="font-semibold">
                {memory(d.bytes, d.device === "cpu" ? 2 : 1)}
              </b>
            </span>
          ))}
          {!devices.length && (
            <span class="text-faint">Placement unavailable</span>
          )}
        </span>
      </div>
    </div>
  );
}
