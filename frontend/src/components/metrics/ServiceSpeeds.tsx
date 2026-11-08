// Services ranked by their mean output rate over the range, as bars
// against the fastest. The time series shows when a service was fast;
// this shows which services are fast at a glance.

import { useTranslation } from "react-i18next";

import { formatTokenRate } from "../../util.ts";
import { Panel } from "../ui/layout/Panel.tsx";
import type { ServiceSpeed } from "./metricsViewData.ts";

export function ServiceSpeeds({
  speeds,
}: {
  readonly speeds: readonly ServiceSpeed[];
}) {
  const { t } = useTranslation();
  const max = Math.max(0, ...speeds.map((s) => s.tokensPerSecond));
  return (
    <Panel heading={t("stats.serviceSpeeds")} padding="none">
      {speeds.length === 0 ? (
        <p className="px-4 py-3 text-sm text-muted">
          {t("stats.noServiceSpeeds")}
        </p>
      ) : (
        <ol className="divide-y divide-line-subtle">
          {speeds.map((s) => {
            const rate = `${formatTokenRate(s.tokensPerSecond)} tok/s`;
            const label = s.effective
              ? t("stats.speedEffective", { rate })
              : rate;
            return (
              <li
                key={s.service}
                className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 px-4 py-2 sm:grid-cols-[minmax(0,14rem)_minmax(0,1fr)_7rem]"
              >
                <span className="min-w-0 font-mono text-xs wrap-anywhere text-ink">
                  {s.service}
                </span>
                <div
                  role="img"
                  aria-label={t("stats.speedBar", {
                    service: s.service,
                    rate: label,
                  })}
                  className="col-span-full row-start-2 h-3 border border-line-strong bg-canvas sm:col-span-1 sm:col-start-2 sm:row-start-1"
                >
                  <div
                    className="h-full bg-accent-data"
                    style={{
                      width: `${max > 0 ? (s.tokensPerSecond / max) * 100 : 0}%`,
                    }}
                  />
                </div>
                <span className="text-right font-mono text-xs text-soft tabular-nums">
                  {label}
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}
