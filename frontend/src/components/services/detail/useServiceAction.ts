// Runs lifecycle actions on one service and keeps the last failure to
// show beside the control that caused it, including the 202 answers that
// mean a start did not happen. The next action clears it.

import { useState } from "react";
import { useTranslation } from "react-i18next";

import { useLifecycle, type LifecycleAction } from "../../../api/hooks.ts";
import { actionOutcome } from "./serviceActions.ts";

export type ServiceAction = {
  readonly run: (action: LifecycleAction) => void;
  readonly pending: boolean;
  readonly error: string | null;
};

export function useServiceAction(name: string): ServiceAction {
  const { t } = useTranslation();
  const lifecycle = useLifecycle();
  const [error, setError] = useState<string | null>(null);

  const run = (action: LifecycleAction) => {
    setError(null);
    lifecycle.mutateAsync({ action, name }).then(
      (response) => {
        const outcome = actionOutcome(response);
        if (outcome.kind === "error") {
          setError(t("services.actionFailed", { message: outcome.message }));
        } else if (outcome.kind === "queue_full") {
          setError(t("services.queueFull"));
        }
      },
      (e: unknown) =>
        setError(
          t("services.actionFailed", {
            message: e instanceof Error ? e.message : String(e),
          }),
        ),
    );
  };

  return { run, pending: lifecycle.isPending, error };
}
