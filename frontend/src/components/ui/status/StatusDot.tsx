// A small filled dot in a tone's colour. Pass `state` (and a disabled
// service's `reason`) to derive the tone from a service state, or `tone`
// directly.

import { useTranslation } from "react-i18next";

import type { DisableReason, ServiceStateKind } from "../../../api/client.ts";
import { cn } from "../cn.ts";
import { toneText, type Tone } from "../tone.ts";
import { reasonLabelKey, stateLabelKey, stateTone } from "./stateTone.ts";

type StatusDotProps = (
  | {
      readonly state: ServiceStateKind;
      readonly reason?: DisableReason | null;
      readonly tone?: never;
    }
  | { readonly tone: Tone; readonly state?: never; readonly reason?: never }
) & { readonly className?: string };

export function StatusDot({ state, reason, tone, className }: StatusDotProps) {
  const { t } = useTranslation();
  const resolved = state === undefined ? tone : stateTone(state, reason);
  const label =
    state === undefined
      ? undefined
      : [
          t(stateLabelKey(state), { defaultValue: state }),
          reason == null
            ? undefined
            : t(reasonLabelKey(reason), { defaultValue: reason }),
        ]
          .filter((part) => part !== undefined)
          .join(" · ");
  return (
    <span
      className={cn(
        "inline-block size-2 shrink-0 rounded-full bg-current",
        toneText({ tone: resolved }),
        className,
      )}
      title={label}
    />
  );
}
