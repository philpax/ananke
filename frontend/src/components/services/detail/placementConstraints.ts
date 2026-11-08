// The placement constraints a service's config sets, as label/value rows
// for the Placement panel.

import type { TFunction } from "i18next";

import type { PlacementConfig } from "../../../api/client.ts";
import { formatBytes, formatDuration } from "../../../util.ts";

export type ConstraintRow = {
  readonly label: string;
  readonly value: string;
  readonly mono?: boolean;
};

export function constraintRows(
  placement: PlacementConfig,
  t: TFunction,
): ConstraintRow[] {
  const gpus = placement.gpu_allow ?? [];
  const rows: ConstraintRow[] = [
    {
      label: t("services.placement.policy"),
      value: t(`services.placement.policies.${placement.policy}`),
    },
    {
      label: t("services.placement.gpus"),
      value:
        gpus.length === 0 ? t("services.placement.allGpus") : gpus.join(", "),
      mono: gpus.length > 0,
    },
  ];
  if (placement.split) {
    rows.push({
      label: t("services.placement.split"),
      value: t(`services.placement.splitModes.${placement.split.mode}`),
    });
    const weights = placement.split.tensor_split_weights;
    if (weights && weights.length > 0) {
      rows.push({
        label: t("services.placement.weights"),
        // Weights are one per allowed GPU in ascending id order; with no
        // allow-list they are one per GPU from 0.
        value: weights
          .map((w, i) => `${[...gpus].sort((a, b) => a - b)[i] ?? i}: ${w}`)
          .join(" · "),
        mono: true,
      });
    }
  }
  if (placement.gpu_headroom_bytes > 0) {
    rows.push({
      label: t("services.placement.headroom"),
      value: formatBytes(placement.gpu_headroom_bytes),
      mono: true,
    });
  }
  rows.push(allocationRow(placement.allocation, t));
  return rows;
}

function allocationRow(
  allocation: PlacementConfig["allocation"],
  t: TFunction,
): ConstraintRow {
  const label = t("services.placement.allocation");
  switch (allocation.kind) {
    case "estimated":
      return { label, value: t("services.placement.estimated") };
    case "static":
      return {
        label,
        value: t("services.placement.static", {
          bytes: formatBytes(allocation.reserve_bytes),
        }),
      };
    case "dynamic":
      return {
        label,
        value: t("services.placement.dynamic", {
          min: formatBytes(allocation.min_bytes),
          max: formatBytes(allocation.max_bytes),
          runtime: formatDuration(allocation.min_borrower_runtime_ms),
        }),
      };
    default: {
      const unreachable: never = allocation;
      return unreachable;
    }
  }
}
