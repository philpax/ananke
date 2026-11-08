// The shared time-range control: rolling presets plus a fixed range. It is
// controlled, and its value lives in the URL (see `timeRange.ts`), so every
// view that shows it shows the same range. Picking "custom" starts from the
// last hour; a fixed range applies once both ends are valid and in order.

import { useTranslation } from "react-i18next";

import { SegmentedToggle } from "./SegmentedToggle.tsx";
import { TextInput } from "./TextInput.tsx";
import {
  RANGE_PRESETS,
  datetimeToMs,
  msToDatetime,
  type RangePreset,
  type TimeRange,
} from "./timeRange.ts";

export function TimeRangeSelect({
  value,
  onChange,
}: {
  readonly value: TimeRange;
  readonly onChange: (range: TimeRange) => void;
}) {
  const { t } = useTranslation();

  function setBound(bound: "from" | "to", input: string) {
    if (value.kind !== "fixed") return;
    const ms = datetimeToMs(input);
    if (ms === null) return;
    const next = { ...value, [bound === "from" ? "fromMs" : "toMs"]: ms };
    if (next.fromMs < next.toMs) onChange(next);
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <SegmentedToggle<RangePreset | "custom">
        label={t("common.timeRange")}
        options={[
          ...RANGE_PRESETS.map((p) => ({ label: p.id, value: p.id })),
          { label: t("common.custom"), value: "custom" },
        ]}
        selected={value.kind === "fixed" ? "custom" : value.preset}
        onChange={(selected) => {
          if (selected !== "custom") {
            onChange({ kind: "rolling", preset: selected });
          } else if (value.kind !== "fixed") {
            const toMs = Math.ceil(Date.now() / MINUTE_MS) * MINUTE_MS;
            onChange({ kind: "fixed", fromMs: toMs - HOUR_MS, toMs });
          }
        }}
      />
      {value.kind === "fixed" && (
        <>
          <label className="flex items-center gap-1.5 text-xs text-muted">
            {t("common.from")}
            <TextInput
              type="datetime-local"
              size="sm"
              mono
              className="px-1.5 text-xs"
              value={msToDatetime(value.fromMs)}
              max={msToDatetime(value.toMs)}
              onChange={(e) => setBound("from", e.target.value)}
            />
          </label>
          <label className="flex items-center gap-1.5 text-xs text-muted">
            {t("common.to")}
            <TextInput
              type="datetime-local"
              size="sm"
              mono
              className="px-1.5 text-xs"
              value={msToDatetime(value.toMs)}
              min={msToDatetime(value.fromMs)}
              onChange={(e) => setBound("to", e.target.value)}
            />
          </label>
        </>
      )}
    </div>
  );
}

const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;
