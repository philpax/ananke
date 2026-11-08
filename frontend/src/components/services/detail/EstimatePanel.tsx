// The memory estimate's parts, the peak actually observed, and how far the
// rolling correction has moved the estimate from what the model predicts.

import { useTranslation } from "react-i18next";

import type { EstimateSummary, ServiceDetail } from "../../../api/client.ts";
import { formatBytes } from "../../../util.ts";
import { FactList } from "../../ui/data/FactList.tsx";
import { Panel } from "../../ui/layout/Panel.tsx";

export function EstimatePanel({
  estimate,
  detail,
}: {
  readonly estimate: EstimateSummary;
  readonly detail: ServiceDetail;
}) {
  const { t } = useTranslation();
  const total =
    estimate.weights_bytes +
    estimate.kv_bytes_for_context +
    estimate.compute_buffer_bytes_per_device;
  const drift = (mean: number | null | undefined, samples: number) =>
    mean != null &&
    t("serviceDetail.driftValue", {
      value: mean.toFixed(3),
      count: samples,
    });
  const vramDrift = drift(detail.rolling_mean, detail.rolling_samples);
  const hostDrift = drift(
    detail.rolling_mean_host,
    detail.rolling_samples_host,
  );
  return (
    <Panel heading={t("serviceDetail.memoryEstimate")}>
      <FactList
        facts={[
          {
            label: t("serviceDetail.weights"),
            value: formatBytes(estimate.weights_bytes),
            mono: true,
          },
          {
            label: t("serviceDetail.kvAtContext", {
              ctx: estimate.configured_context.toLocaleString(),
            }),
            value: formatBytes(estimate.kv_bytes_for_context),
            mono: true,
          },
          {
            label: t("serviceDetail.computeDev"),
            value: formatBytes(estimate.compute_buffer_bytes_per_device),
            mono: true,
          },
          {
            label: t("serviceDetail.total"),
            value: formatBytes(total),
            mono: true,
          },
          detail.observed_peak_bytes > 0 && {
            label: t("serviceDetail.observedPeak"),
            value: formatBytes(detail.observed_peak_bytes),
            mono: true,
          },
          vramDrift && {
            label: t("serviceDetail.estimatorDrift"),
            value: vramDrift,
            mono: true,
          },
          hostDrift && {
            label: t("serviceDetail.estimatorDriftHost"),
            value: hostDrift,
            mono: true,
          },
        ]}
      />
    </Panel>
  );
}
