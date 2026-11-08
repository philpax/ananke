// A service state as a dot and a translated label in the state's tone. A
// disabled service's reason follows the state: "disabled · OOM".

import { useTranslation } from "react-i18next";

import type { DisableReason, ServiceStateKind } from "../../../api/client.ts";
import { cn } from "../cn.ts";
import { toneBox } from "../tone.ts";
import { reasonLabelKey, stateLabelKey, stateTone } from "./stateTone.ts";

export function StateChip({
  state,
  reason,
  className,
}: {
  readonly state: ServiceStateKind;
  readonly reason?: DisableReason | null;
  readonly className?: string;
}) {
  const { t } = useTranslation();
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-semibold",
        toneBox({ tone: stateTone(state, reason) }),
        className,
      )}
    >
      <span aria-hidden="true" className="size-1.5 rounded-full bg-current" />
      {t(stateLabelKey(state), { defaultValue: state })}
      {reason != null && (
        <span className="font-normal">
          <span aria-hidden="true">·</span>{" "}
          {t(reasonLabelKey(reason), { defaultValue: reason })}
        </span>
      )}
    </span>
  );
}
