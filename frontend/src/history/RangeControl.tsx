import clsx from "clsx";
import { twMerge } from "tailwind-merge";
import { useEffect, useState } from "preact/hooks";

import type { Range } from "../api/client";
import { localDate, parseRange, presetRange } from "./range";

export function RangeControl({
  range,
  onChange,
}: {
  readonly range: Range;
  readonly onChange: (range: Range) => void;
}) {
  const [from, setFrom] = useState(localDate(range.since)),
    [to, setTo] = useState(localDate(range.until)),
    [error, setError] = useState("");
  useEffect(() => {
    setFrom(localDate(range.since));
    setTo(localDate(range.until));
  }, [range.since, range.until]);
  const button =
    "h-8 border border-line-strong bg-surface px-2.5 text-xs text-body";
  const input =
    "h-8 max-w-[184px] border border-line-strong bg-canvas px-1.5 font-mono text-xs text-ink [color-scheme:dark]";
  return (
    <div
      role="group"
      aria-label="Time range"
      class="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2"
    >
      <div class="flex gap-0.5" aria-label="Preset ranges">
        {[1, 6, 24].map((hours) => (
          <button
            type="button"
            key={hours}
            aria-pressed={range.preset === hours}
            class={twMerge(
              clsx(
                button,
                range.preset === hours &&
                  `border-accent bg-selected text-accent-text`,
              ),
            )}
            onClick={() => {
              setError("");
              onChange(presetRange(hours));
            }}
          >
            {hours}h
          </button>
        ))}
      </div>
      <label class="flex items-center gap-[5px] text-xs text-muted">
        From{" "}
        <input
          class={input}
          aria-label="From date and time"
          type="datetime-local"
          value={from}
          onInput={(e) => setFrom(e.currentTarget.value)}
        />
      </label>
      <label class="flex items-center gap-[5px] text-xs text-muted">
        To{" "}
        <input
          class={input}
          aria-label="To date and time"
          type="datetime-local"
          value={to}
          onInput={(e) => setTo(e.currentTarget.value)}
        />
      </label>
      <button
        type="button"
        class={twMerge(clsx(button, `bg-raised text-ink`))}
        onClick={() => {
          const result = parseRange(from, to);
          if (typeof result === "string") setError(result);
          else {
            setError("");
            onChange(result);
          }
        }}
      >
        Apply
      </button>
      {error && (
        <span role="status" class="text-xs text-danger">
          {error}
        </span>
      )}
    </div>
  );
}
