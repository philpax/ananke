import type { ValidateFunction } from "ajv";

import { codec, deviceListCodec } from "./contract";
import type { Schemas } from "./contract";
import { command, read } from "./socket";
import type { Source } from "./socket";

export type Result<T> =
  | { readonly kind: "ok"; readonly value: T }
  | { readonly kind: "error"; readonly message: string };
export type Range = {
  readonly since: number;
  readonly until: number;
  readonly preset: number | null;
};

function source<T>(
  query: Schemas["DashboardQuery"],
  validate: ValidateFunction<T>,
): Source<T> {
  return { query, validate };
}
const config = source({ type: "config" }, codec("ConfigResponse"));
const validation = codec("ConfigValidateResponse");

export const api = {
  services: source({ type: "services" }, codec("ServicesResponse")),
  detail: (name: string) =>
    source({ type: "service", name }, codec("ServiceDetail")),
  devices: source({ type: "devices" }, deviceListCodec),
  info: source({ type: "info" }, codec("DaemonInfoResponse")),
  events: source({ type: "events" }, codec("DashboardEvents")),
  models: source({ type: "models" }, codec("ModelsResponse")),
  metrics: (range: Range, service?: string) =>
    source(
      {
        type: "metrics",
        range: window(range),
        service: service ?? null,
        bucket: bucketSize(range).label,
      },
      codec("MetricsResponse"),
    ),
  restarts: (range: Range, service?: string) =>
    source(
      { type: "restarts", range: window(range), service: service ?? null },
      codec("RestartsResponse"),
    ),
  samples: (range: Range) =>
    source(
      { type: "samples", range: window(range) },
      codec("DeviceSamplesResponse"),
    ),
  logSource: (name: string, range: Range, before?: string) =>
    source(
      { type: "logs", name, range: window(range), before: before ?? null },
      codec("LogsResponse"),
    ),
  logs: (name: string, range: Range, before?: string, signal?: AbortSignal) =>
    read(api.logSource(name, range, before), signal),
  config: (signal?: AbortSignal) => read(config, signal),
  validate: async (content: string) => {
    const result = await command({ type: "validate", content });
    if (result.kind === "error") return result;
    return validation(result.value)
      ? { kind: "ok" as const, value: result.value }
      : {
          kind: "error" as const,
          message: "Config: the server returns invalid validation results.",
        };
  },
  save: async (content: string, hash: string): Promise<Result<void>> => {
    const result = await command({ type: "save", content, hash });
    return result.kind === "error" ? result : { kind: "ok", value: undefined };
  },
  action: async (
    name: string,
    action: Schemas["DashboardAction"],
  ): Promise<Result<void>> => {
    const result = await command({ type: "action", name, action });
    return result.kind === "error" ? result : { kind: "ok", value: undefined };
  },
};

export function bucketSize(range: Range) {
  const span = range.until - range.since;
  return span <= 3_600_000
    ? { label: "1m", ms: 60_000 }
    : span <= 86_400_000
      ? { label: "5m", ms: 300_000 }
      : { label: "1h", ms: 3_600_000 };
}

export function rangeKey(range: Range) {
  return range.preset
    ? `live:${range.preset}`
    : `${range.since}:${range.until}`;
}

function window(range: Range): Schemas["DashboardRange"] {
  return {
    since: range.since,
    until: range.until,
    live_ms: range.preset ? range.preset * 3_600_000 : null,
  };
}
