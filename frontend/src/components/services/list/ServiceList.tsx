// The service list: search, sort, and a favourites filter over services
// grouped by state. The rows share one tab stop; the arrow keys, Home, and
// End move it, and Tab from a row reaches that row's own controls.

import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { useLocation } from "react-router-dom";

import { toggleFavourite, useFavourites } from "../../../api/favourites.ts";
import { useServices } from "../../../api/hooks.ts";
import { useStoredValue } from "../../../hooks/useStoredValue.ts";
import { Button } from "../../ui/button/Button.tsx";
import { Message } from "../../ui/feedback/Message.tsx";
import { Checkbox } from "../../ui/form/Checkbox.tsx";
import { SegmentedToggle } from "../../ui/form/SegmentedToggle.tsx";
import { TextInput } from "../../ui/form/TextInput.tsx";
import { rangeSearch } from "../../ui/form/timeRange.ts";
import { GroupHeader } from "../../ui/layout/GroupHeader.tsx";
import { stateLabelKey, stateTone } from "../../ui/status/stateTone.ts";
import { ServiceListRow } from "./ServiceListRow.tsx";
import {
  SORT_ORDERS,
  filterServices,
  groupByState,
  parseSortOrder,
  sortServices,
} from "./serviceList.ts";
import { useServiceSparklines } from "./useServiceSparklines.ts";

export function ServiceList({
  selected,
  onHide,
}: {
  readonly selected: string | undefined;
  readonly onHide: () => void;
}) {
  const { t } = useTranslation();
  const { search } = useLocation();
  const services = useServices();
  const favourites = useFavourites();
  const spark = useServiceSparklines();
  const [query, setQuery] = useState("");
  const [favouritesOnly, setFavouritesOnly] = useState(false);
  const [order, setOrder] = useStoredValue(SORT_KEY, parseSortOrder);
  const [focused, setFocused] = useState<string | undefined>(undefined);

  const all = services.data ?? [];
  const shown = sortServices(
    filterServices(all, { query, favouritesOnly, favourites }),
    order,
    favourites,
  );
  const groups = groupByState(shown);
  const flat = groups.flatMap((g) => g.services.map((s) => s.name));
  const current =
    [focused, selected].find((n) => n !== undefined && flat.includes(n)) ??
    flat[0];
  const range = rangeSearch(search);

  // A deep link lands on a service that may be far down the list.
  const selectedRef = useRef<HTMLAnchorElement>(null);
  useEffect(() => {
    selectedRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [selected]);

  const listRef = useRef<HTMLDivElement>(null);
  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    const index = current === undefined ? -1 : flat.indexOf(current);
    const target =
      e.key === "ArrowDown"
        ? flat[Math.min(index + 1, flat.length - 1)]
        : e.key === "ArrowUp"
          ? flat[Math.max(index - 1, 0)]
          : e.key === "Home"
            ? flat[0]
            : e.key === "End"
              ? flat.at(-1)
              : undefined;
    if (target === undefined) return;
    if (!(e.target instanceof HTMLAnchorElement)) return;
    e.preventDefault();
    setFocused(target);
    const links = listRef.current?.querySelectorAll<HTMLAnchorElement>(
      "a[data-service-link]",
    );
    links?.[flat.indexOf(target)]?.focus();
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-col gap-2 border-b border-line px-3 py-2.5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-ink">
            {t("services.title")}
          </h2>
          <p role="status" className="font-mono text-xs text-muted">
            {t("services.list.shown", {
              shown: shown.length,
              total: all.length,
            })}
          </p>
          <Button
            tone="quiet"
            size="icon-sm"
            icon="hide"
            aria-label={t("services.list.hide")}
            title={t("services.list.hide")}
            onClick={onHide}
            className="ml-auto max-lg:hidden"
          />
        </div>
        <TextInput
          type="search"
          size="sm"
          aria-label={t("services.list.search")}
          placeholder={t("services.list.searchPlaceholder")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <SegmentedToggle
            label={t("services.list.sort")}
            options={SORT_ORDERS.map((o) => ({
              label: t(`services.list.sortOrders.${o}`),
              value: o,
            }))}
            selected={order}
            onChange={setOrder}
          />
          <Checkbox
            label={t("services.list.favouritesOnly")}
            checked={favouritesOnly}
            onChange={setFavouritesOnly}
            className="text-xs"
          />
        </div>
      </div>
      <div
        ref={listRef}
        onKeyDown={onKeyDown}
        className="min-h-0 flex-1 overflow-y-auto"
      >
        {services.isPending && <Message kind="loading" className="px-3" />}
        {services.error && (
          <Message
            kind="error"
            text={services.error.message}
            className="px-3"
          />
        )}
        {services.data && shown.length === 0 && (
          <Message
            kind="empty"
            text={t("services.list.noMatch")}
            className="px-3"
          />
        )}
        {groups.map((group) => {
          const label = t(stateLabelKey(group.state));
          return (
            <div key={group.state}>
              <GroupHeader
                tone={stateTone(group.state)}
                count={group.services.length}
              >
                {label}
              </GroupHeader>
              <ul aria-label={label}>
                {group.services.map((s) => (
                  <ServiceListRow
                    key={s.name}
                    service={s}
                    href={`/services/${encodeURIComponent(s.name)}${range}`}
                    selected={s.name === selected}
                    current={s.name === current}
                    favourite={favourites.has(s.name)}
                    onFavourite={() => toggleFavourite(s.name)}
                    spark={spark.byService.get(s.name)}
                    xMin={spark.xMin}
                    xMax={spark.xMax}
                    linkRef={s.name === selected ? selectedRef : undefined}
                  />
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const SORT_KEY = "ananke-sort-order";
