// The totals over the range as tiles: how many services run, then
// requests, errors, tokens, latency, and token rates. With one service
// picked, the tiles are that service's and the running count drops out.

import { useTranslation } from "react-i18next";

import { formatTokenRate } from "../../util.ts";
import type { HistorySummary } from "../services/detail/history/historySummary.ts";
import { cn } from "../ui/cn.ts";
import { Stat } from "../ui/data/Stat.tsx";

export function FleetSummary({
  summary,
  services,
  loading,
}: {
  readonly summary: HistorySummary;
  /// The fleet's running and total counts; omitted with a service picked.
  readonly services?: { readonly running: number; readonly total: number };
  /// Metrics have not arrived, so the range totals are not known yet.
  readonly loading: boolean;
}) {
  const { t } = useTranslation();
  const known = (value: string) => (loading ? "—" : value);
  const rate = (v: number | null) => (v === null ? "—" : formatTokenRate(v));
  return (
    <section
      aria-label={t("stats.totals")}
      className="grid grid-cols-[repeat(auto-fill,minmax(7rem,1fr))] gap-4 border border-line bg-surface p-4"
    >
      {services && (
        <Stat
          size="xl"
          label={t("stats.runningServices")}
          value={`${services.running} / ${services.total}`}
        />
      )}
      <Stat
        size="xl"
        label={t("stats.requests")}
        value={known(summary.requests.toLocaleString())}
      />
      <Stat
        size="xl"
        label={t("stats.errors")}
        value={
          <span className={cn(!loading && summary.errors > 0 && "text-danger")}>
            {known(summary.errors.toLocaleString())}
          </span>
        }
      />
      <Stat
        size="xl"
        label={t("stats.tokensIn")}
        value={known(summary.inputTokens.toLocaleString())}
      />
      <Stat
        size="xl"
        label={t("stats.tokensOut")}
        value={known(summary.outputTokens.toLocaleString())}
      />
      <Stat
        size="xl"
        label={t("stats.avgLatency")}
        value={known(
          summary.avgLatencyMs === null
            ? "—"
            : Math.round(summary.avgLatencyMs).toLocaleString(),
        )}
        unit={summary.avgLatencyMs === null || loading ? undefined : "ms"}
      />
      {summary.hasSplitTps && (
        <>
          <Stat
            size="xl"
            label={t("stats.avgTpsIn")}
            value={known(rate(summary.avgInputTps))}
          />
          <Stat
            size="xl"
            label={t("stats.avgTpsOut")}
            value={known(rate(summary.avgOutputTps))}
          />
        </>
      )}
      <Stat
        size="xl"
        label={t("stats.avgTpsEffective")}
        value={known(rate(summary.avgEffectiveTps))}
      />
    </section>
  );
}
