// State badge for service states. Maps each state to a semantic colour
// variant. Used across the dashboard, services table, and detail view.

import { Badge } from "./Badge.tsx";
import { useTranslation } from "react-i18next";

type StateBadgeProps = {
  state: string;
};

type Variant = "success" | "warning" | "danger" | "neutral";

function stateVariant(state: string): Variant {
  if (state === "running") return "success";
  if (state === "starting") return "warning";
  if (state === "draining") return "warning";
  if (state === "failed") return "danger";
  if (state.startsWith("disabled")) return "neutral";
  return "neutral";
}

function stateKey(state: string): string {
  // `disabled` may carry a suffix (e.g. `disabled_no_quota`); map
  // them all to the base `disabled` translation key.
  const base = state.startsWith("disabled") ? "disabled" : state;
  return `services.states.${base}`;
}

export function StateBadge({ state }: StateBadgeProps) {
  const { t } = useTranslation();
  return (
    <Badge variant={stateVariant(state)}>
      {t(stateKey(state), { defaultValue: state })}
    </Badge>
  );
}
