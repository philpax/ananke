// Chart colours by token name. Callers pick a `SeriesColor`; the chart
// resolves it against the live theme, and the sparkline maps it to
// utility classes, so no component holds a hex value.

export type SeriesColor =
  | "series-1"
  | "series-2"
  | "series-3"
  | "series-4"
  | "series-5"
  | "series-6"
  | "device-cpu"
  | "ok"
  | "warn"
  | "danger";

const CATEGORICAL: readonly SeriesColor[] = [
  "series-1",
  "series-2",
  "series-3",
  "series-4",
  "series-5",
  "series-6",
];

/// The `i`th colour of the categorical set, wrapping after six.
export function categorical(i: number): SeriesColor {
  return CATEGORICAL[i % CATEGORICAL.length] ?? "series-1";
}

/// A stable colour per device: `gpu:N` cycles through the first five
/// series, so GPU 0 is the accent; the CPU has its own neutral.
export function deviceSeries(id: string): SeriesColor {
  const gpu = /^gpu:(\d+)$/.exec(id);
  if (gpu?.[1] !== undefined) return categorical(Number(gpu[1]) % 5);
  return "device-cpu";
}

/// The token for a colour's translucent area fill. Status colours use the
/// tone's background, which is already a quiet tint.
export function fillToken(color: SeriesColor): string {
  switch (color) {
    case "ok":
    case "warn":
    case "danger":
      return `${color}-bg`;
    case "device-cpu":
      return "series-6-fill";
    default:
      return `${color}-fill`;
  }
}

/// Literal class lookups, so Tailwind sees every class it must generate.
export const seriesBgClassName: Readonly<Record<SeriesColor, string>> = {
  "series-1": "bg-series-1",
  "series-2": "bg-series-2",
  "series-3": "bg-series-3",
  "series-4": "bg-series-4",
  "series-5": "bg-series-5",
  "series-6": "bg-series-6",
  "device-cpu": "bg-device-cpu",
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
};

export const seriesStrokeClassName: Readonly<Record<SeriesColor, string>> = {
  "series-1": "stroke-series-1",
  "series-2": "stroke-series-2",
  "series-3": "stroke-series-3",
  "series-4": "stroke-series-4",
  "series-5": "stroke-series-5",
  "series-6": "stroke-series-6",
  "device-cpu": "stroke-device-cpu",
  ok: "stroke-ok",
  warn: "stroke-warn",
  danger: "stroke-danger",
};

export const seriesFillClassName: Readonly<Record<SeriesColor, string>> = {
  "series-1": "fill-series-1-fill",
  "series-2": "fill-series-2-fill",
  "series-3": "fill-series-3-fill",
  "series-4": "fill-series-4-fill",
  "series-5": "fill-series-5-fill",
  "series-6": "fill-series-6-fill",
  "device-cpu": "fill-series-6-fill",
  ok: "fill-ok-bg",
  warn: "fill-warn-bg",
  danger: "fill-danger-bg",
};
