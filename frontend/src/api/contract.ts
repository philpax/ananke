import Ajv from "ajv/dist/2020.js";

import type { components } from "./types";
import definitions from "./schema.json";

export type Schemas = components["schemas"];
export type Service = Schemas["ServiceSummary"];
export type Detail = Schemas["ServiceDetail"];
export type Device = Schemas["DeviceSummary"];
export type SystemEvent = Schemas["Event"];
export type LogLine = Schemas["LogLine"];
export type Bucket = Schemas["MetricBucketResponse"];
export type Restart = Schemas["ServiceRestartEntry"];
export type Sample = Schemas["DeviceSampleResponse"];
export type Info = Schemas["DaemonInfoResponse"];
export type Config = Schemas["ConfigResponse"];

const ajv = new Ajv({ strict: false, allErrors: true });
ajv.addFormat("int32", { type: "number", validate: Number.isInteger });
ajv.addFormat("int64", { type: "number", validate: Number.isInteger });
ajv.addFormat("float", { type: "number", validate: Number.isFinite });
ajv.addFormat("double", { type: "number", validate: Number.isFinite });
const defs: unknown = JSON.parse(
  JSON.stringify(definitions).replaceAll("#/components/schemas/", "#/$defs/"),
);
ajv.addSchema({ $id: "ananke", $defs: defs });

export function codec<K extends keyof Schemas>(name: K) {
  return ajv.compile<Schemas[K]>({ $ref: `ananke#/$defs/${name}` });
}

export const eventCodec = codec("Event");
export const logCodec = codec("LogStreamMessage");
export const chunkCodec = codec("ChatCompletionChunk");
export const deviceListCodec = ajv.compile<Device[]>({
  type: "array",
  items: { $ref: "ananke#/$defs/DeviceSummary" },
});
