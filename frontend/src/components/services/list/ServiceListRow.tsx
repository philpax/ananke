// One service in the list: favourite star, name, state, footprint, tags,
// a request sparkline, and a start/stop button. The name is a link
// stretched over the row; the controls sit above it as siblings, so no
// interactive element nests inside the link.

import { useId, type Ref } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import type { ServiceSummary } from "../../../api/client.ts";
import { formatBytes } from "../../../util.ts";
import { Button } from "../../ui/button/Button.tsx";
import { cn } from "../../ui/cn.ts";
import { Sparkline } from "../../ui/data/Sparkline.tsx";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { Icon } from "../../ui/icon/Icon.tsx";
import { rowVariants } from "../../ui/layout/rowVariants.ts";
import { Chip } from "../../ui/status/Chip.tsx";
import { StatusDot } from "../../ui/status/StatusDot.tsx";
import { stateTone } from "../../ui/status/stateTone.ts";
import { availableActions } from "../detail/serviceActions.ts";
import { useServiceAction } from "../detail/useServiceAction.ts";
import { RowFootprint } from "./RowFootprint.tsx";

export function ServiceListRow({
  service,
  href,
  selected,
  current,
  favourite,
  onFavourite,
  spark,
  xMin,
  xMax,
  linkRef,
}: {
  readonly service: ServiceSummary;
  readonly href: string;
  readonly selected: boolean;
  /// The row holding the list's single tab stop.
  readonly current: boolean;
  readonly favourite: boolean;
  readonly onFavourite: () => void;
  readonly spark: { ts: number[]; counts: number[] } | undefined;
  readonly xMin: number;
  readonly xMax: number;
  readonly linkRef?: Ref<HTMLAnchorElement>;
}) {
  const { t } = useTranslation();
  const action = useServiceAction(service.name);
  const toggle = availableActions(service.state).toggle;
  const tabIndex = current ? 0 : -1;
  const disabled = service.state === "disabled";
  const verdict = disabled ? undefined : service.fit_verdict?.kind;
  const verdictId = useId();
  const shortfalls =
    service.fit_verdict?.kind === "does_not_fit"
      ? service.fit_verdict.shortfalls
          .map((s) =>
            t("serviceDetail.shortfall", {
              device: s.device,
              requested: formatBytes(s.requested_bytes),
              available: formatBytes(s.available_bytes),
            }),
          )
          .join("; ")
      : "";
  const inflight = service.inflight_count ?? 0;

  return (
    <li
      className={cn(
        "group relative grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-2 border-b border-line-subtle px-2 py-1.5",
        rowVariants({
          selected,
          severity:
            stateTone(service.state, service.disabled_reason) === "danger"
              ? "danger"
              : "neutral",
        }),
      )}
    >
      <button
        type="button"
        aria-pressed={favourite}
        aria-label={t("services.list.favourite", { name: service.name })}
        tabIndex={tabIndex}
        onClick={onFavourite}
        className={cn(
          "relative z-10 inline-flex size-6 items-center justify-center pointer-coarse:size-touch",
          favourite ? "text-warn" : "text-faint hover:text-soft",
        )}
      >
        <Icon name="star" size="3.5" filled={favourite} />
      </button>
      <div className="flex min-w-0 flex-col gap-0.5">
        <div className="flex min-w-0 items-center gap-1.5">
          <Link
            ref={linkRef}
            to={href}
            tabIndex={tabIndex}
            aria-current={selected ? "page" : undefined}
            aria-describedby={verdict ? verdictId : undefined}
            data-service-link=""
            title={service.description ?? undefined}
            className={cn(
              "min-w-0 font-mono text-sm wrap-anywhere after:absolute after:inset-0",
              verdict === "does_not_fit"
                ? "text-danger"
                : verdict === "needs_eviction"
                  ? "text-warn"
                  : "text-ink",
            )}
          >
            {service.name}
          </Link>
          <StatusDot state={service.state} reason={service.disabled_reason} />
          <CopyButton
            value={service.name}
            tabIndex={tabIndex}
            className="relative z-10 opacity-0 group-focus-within:opacity-100 group-hover:opacity-100 pointer-coarse:opacity-100"
          />
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5">
          <RowFootprint service={service} />
          {service.has_mmproj && <Chip tone="vision">vision</Chip>}
          {service.modality === "embedding" && (
            <Chip tone="embedding">embedding</Chip>
          )}
          {inflight > 0 && (
            <Chip tone="accent">
              {t("services.list.inflight", { count: inflight })}
            </Chip>
          )}
          {disabled && service.disabled_reason && (
            <Chip
              tone={
                stateTone("disabled", service.disabled_reason) === "danger"
                  ? "danger"
                  : "neutral"
              }
            >
              {t(`services.disabledReasons.${service.disabled_reason}`)}
            </Chip>
          )}
          {verdict === "does_not_fit" && (
            <Chip tone="danger">{t("services.list.wontFit")}</Chip>
          )}
          {verdict === "needs_eviction" && (
            <Chip tone="warn">{t("serviceDetail.needsEviction")}</Chip>
          )}
          <div className="ml-auto h-5 w-16 shrink-0 @max-list-spark/list:hidden">
            <Sparkline
              data={spark ? [spark.ts, spark.counts] : [[], []]}
              color="series-1"
              height={20}
              xMin={xMin}
              xMax={xMax}
            />
          </div>
          {verdict && (
            <span id={verdictId} className="sr-only">
              {verdict === "does_not_fit"
                ? `${t("serviceDetail.noPlacementFits")} ${shortfalls}`
                : t("serviceDetail.needsEviction")}
            </span>
          )}
        </div>
      </div>
      <div className="relative z-10 flex size-control-sm items-center justify-center pointer-coarse:size-touch">
        {toggle && (
          <Button
            size="icon-sm"
            tone="quiet"
            tabIndex={tabIndex}
            aria-label={t(`services.actions.${toggle}`)}
            title={t(`services.actions.${toggle}`)}
            disabled={action.pending}
            onClick={() => action.run(toggle)}
            className="opacity-0 group-focus-within:opacity-100 group-hover:opacity-100 pointer-coarse:size-touch pointer-coarse:opacity-100"
          >
            <Icon name={toggle} size="3.5" />
          </Button>
        )}
      </div>
      {action.error && (
        <p
          role="alert"
          className="relative z-10 col-span-3 pl-8 text-xs wrap-anywhere text-danger"
        >
          {action.error}
        </p>
      )}
    </li>
  );
}
