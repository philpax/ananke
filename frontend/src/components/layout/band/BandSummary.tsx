// The narrow-screen stand-in for the device cells: "gpu 45% · cpu 30%" as
// a button that opens the cells below the band.

import type { Ref } from "react";
import { useTranslation } from "react-i18next";

import { cn } from "../../ui/cn.ts";
import { Icon } from "../../ui/icon/Icon.tsx";
import { toneText } from "../../ui/tone.ts";
import type { BandSummary as Summary } from "./hardware.ts";

type BandSummaryProps = {
  readonly summary: Summary;
  readonly expanded: boolean;
  readonly onToggle: () => void;
  /// The id of the panel the button opens.
  readonly controls: string;
  readonly ref?: Ref<HTMLButtonElement>;
};

export function BandSummary({
  summary,
  expanded,
  onToggle,
  controls,
  ref,
}: BandSummaryProps) {
  const { t } = useTranslation();
  const { gpu, cpu } = summary;
  return (
    <button
      ref={ref}
      type="button"
      aria-expanded={expanded}
      aria-controls={controls}
      onClick={onToggle}
      className="flex h-touch shrink-0 items-center gap-1.5 border border-line-strong bg-raised px-2.5 font-mono text-xs text-body"
    >
      <span className="sr-only">{t("band.summary.label")}</span>
      {gpu === undefined && cpu === undefined && (
        <span className="text-muted">{t("band.summary.empty")}</span>
      )}
      {gpu !== undefined && (
        <span className={toneText({ tone: gpu.tone })}>
          {gpu.showMax
            ? t("band.summary.gpuWithMax", {
                percent: gpu.mean,
                max: gpu.max,
              })
            : t("band.summary.gpu", { percent: gpu.mean })}
        </span>
      )}
      {gpu !== undefined && cpu !== undefined && (
        <span aria-hidden="true" className="text-faint">
          ·
        </span>
      )}
      {cpu !== undefined && (
        <span className={toneText({ tone: cpu.tone })}>
          {t("band.summary.cpu", { percent: cpu.percent })}
        </span>
      )}
      <Icon
        name="back"
        size="3"
        className={cn("text-muted", expanded ? "rotate-90" : "-rotate-90")}
      />
    </button>
  );
}
