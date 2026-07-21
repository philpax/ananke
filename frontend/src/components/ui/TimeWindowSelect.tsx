// Shared time-window selector: relative presets plus a custom absolute
// range with datetime-local inputs. Extracted from the logs viewer so
// logs, per-service stats, the global stats view, and the dashboard all
// present the same time controls. Model + helpers live in
// ./timeWindow.ts (react-refresh needs this file components-only).

import { useState } from "react";
import { useTranslation } from "react-i18next";

import { SegmentedToggle } from "./SegmentedToggle.tsx";
import {
  TIME_WINDOW_PRESETS,
  datetimeToMs,
  msToDatetime,
  type TimeWindow,
  type TimeWindowPreset,
} from "./timeWindow.ts";

// Matches the logs viewer's original custom-range inputs.
const inputClassName =
  "h-6 rounded-sm border border-border-default bg-base px-1.5 text-xs " +
  "text-primary focus:border-accent focus:outline-none";

export function TimeWindowSelect({
  presets = TIME_WINDOW_PRESETS,
  onChange,
}: {
  presets?: TimeWindowPreset[];
  onChange: (window: TimeWindow) => void;
}) {
  const { t } = useTranslation();
  const [mode, setMode] = useState<"preset" | "custom">("preset");
  const [presetIdx, setPresetIdx] = useState(0);
  const [sinceInput, setSinceInput] = useState("");
  const [untilInput, setUntilInput] = useState("");

  function emitCustom(since: string, until: string) {
    const sinceMs = datetimeToMs(since);
    if (sinceMs === null) return;
    onChange({ kind: "absolute", sinceMs, untilMs: datetimeToMs(until) });
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <SegmentedToggle<number | "custom">
        options={[
          ...presets.map((p, i) => ({ label: p.label, value: i as number })),
          { label: t("common.custom"), value: "custom" as const },
        ]}
        selected={mode === "preset" ? presetIdx : "custom"}
        onChange={(v) => {
          if (v === "custom") {
            if (mode !== "custom") {
              const since = msToDatetime(Date.now() - 60 * 60 * 1000);
              setSinceInput(since);
              setUntilInput("");
              setMode("custom");
              emitCustom(since, "");
            }
          } else {
            setPresetIdx(v);
            setMode("preset");
            onChange({ kind: "relative", durationMs: presets[v]!.durationMs });
          }
        }}
      />
      {mode === "custom" && (
        <>
          <input
            type="datetime-local"
            className={inputClassName}
            value={sinceInput}
            onChange={(e) => {
              setSinceInput(e.target.value);
              emitCustom(e.target.value, untilInput);
            }}
          />
          <span className="text-xs text-tertiary">→</span>
          <input
            type="datetime-local"
            className={inputClassName}
            value={untilInput}
            onChange={(e) => {
              setUntilInput(e.target.value);
              emitCustom(sinceInput, e.target.value);
            }}
          />
        </>
      )}
    </div>
  );
}
