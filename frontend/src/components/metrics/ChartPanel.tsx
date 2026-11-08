// One chart in a headed panel, or a plain "no samples" line in its place
// when the range holds nothing to draw, so an empty range never reads as
// a flat line at zero.

import { useTranslation } from "react-i18next";

import { Chart, type ChartProps } from "../ui/chart/Chart.tsx";
import { Panel } from "../ui/layout/Panel.tsx";

type ChartPanelProps = ChartProps & {
  readonly heading: string;
  readonly empty: boolean;
};

export function ChartPanel({ heading, empty, ...chart }: ChartPanelProps) {
  const { t } = useTranslation();
  return (
    <Panel heading={heading} className="min-w-0">
      {empty ? (
        <p className="flex h-40 items-center justify-center text-sm text-muted">
          {t("stats.noSamples")}
        </p>
      ) : (
        <Chart {...chart} />
      )}
    </Panel>
  );
}
