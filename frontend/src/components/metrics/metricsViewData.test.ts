import { describe, expect, it } from "vitest";

import type {
  DeviceSampleResponse,
  MetricBucketResponse,
} from "../../api/client.ts";
import {
  aggregateBuckets,
  groupByService,
} from "../../api/metrics-aggregate.ts";
import {
  groupDeviceSamples,
  hasSamples,
  hasSplitTps,
  serviceSpeeds,
  splitMemorySeries,
  toFieldsData,
  toMemoryData,
  toMultiSeriesData,
} from "./metricsViewData.ts";

function bucket(
  service: string,
  startMs: number,
  extra: Partial<MetricBucketResponse> = {},
): MetricBucketResponse {
  return {
    service,
    bucket_start: startMs,
    request_count: 1,
    prompt_tokens: 10,
    completion_tokens: 20,
    error_count: 0,
    ...extra,
  };
}

function sample(
  device: string,
  timestampMs: number,
  usedBytes: number,
): DeviceSampleResponse {
  return {
    device,
    timestamp_ms: timestampMs,
    used_bytes: usedBytes,
    free_bytes: 0,
    total_bytes: usedBytes,
  };
}

describe("toFieldsData", () => {
  it("weights rates by request and leaves untimed buckets null", () => {
    const buckets = aggregateBuckets([
      bucket("a", 60_000, { request_count: 1, effective_tps: 10 }),
      bucket("b", 60_000, { request_count: 3, effective_tps: 30 }),
      bucket("a", 120_000),
    ]);
    expect(toFieldsData(buckets, ["requestCount", "effectiveTps"])).toEqual([
      [60, 120],
      [4, 1],
      [25, null],
    ]);
  });

  it("falls back to the effective rate without a split", () => {
    const buckets = aggregateBuckets([bucket("a", 0, { effective_tps: 5 })]);
    expect(hasSplitTps(buckets)).toBe(false);
    expect(
      hasSplitTps(aggregateBuckets([bucket("a", 0, { output_tps: 5 })])),
    ).toBe(true);
  });
});

describe("toMultiSeriesData", () => {
  it("aligns services on the union of their timestamps", () => {
    const perService = groupByService([
      bucket("a", 0, { request_count: 2 }),
      bucket("a", 60_000),
      bucket("b", 60_000, { request_count: 5 }),
    ]);
    expect(perService.map((s) => s.serviceName)).toEqual(["b", "a"]);
    expect(toMultiSeriesData(perService, "requestCount")).toEqual([
      [0, 60],
      [null, 5],
      [2, 1],
    ]);
  });
});

describe("memory", () => {
  const series = groupDeviceSamples([
    sample("gpu:1", 2_000, 3e9),
    sample("cpu", 1_000, 64e9),
    sample("gpu:0", 1_000, 1e9),
    sample("gpu:0", 2_000, 2e9),
  ]);

  it("groups per device in id order and time order", () => {
    expect(series.map((s) => s.device)).toEqual(["cpu", "gpu:0", "gpu:1"]);
    expect(series[1]?.data.map((d) => d.ts)).toEqual([1, 2]);
  });

  it("keeps host memory off the GPU axis", () => {
    const { gpus, host } = splitMemorySeries(series);
    expect(gpus.map((s) => s.device)).toEqual(["gpu:0", "gpu:1"]);
    expect(host.map((s) => s.device)).toEqual(["cpu"]);
    expect(toMemoryData(gpus)).toEqual([
      [1, 2],
      [1, 2],
      [null, 3],
    ]);
  });
});

describe("hasSamples", () => {
  it("ignores the timestamp column", () => {
    expect(hasSamples([[1, 2]])).toBe(false);
    expect(
      hasSamples([
        [1, 2],
        [null, null],
      ]),
    ).toBe(false);
    expect(
      hasSamples([
        [1, 2],
        [null, 0],
      ]),
    ).toBe(true);
  });
});

describe("serviceSpeeds", () => {
  it("ranks by output rate, falling back to the effective rate", () => {
    const speeds = serviceSpeeds(
      groupByService([
        bucket("slow", 0, { output_tps: 10 }),
        bucket("fast", 0, { output_tps: 40, request_count: 1 }),
        bucket("fast", 60_000, { output_tps: 10, request_count: 2 }),
        bucket("plain", 0, { effective_tps: 15 }),
        bucket("idle", 0),
      ]),
    );
    expect(speeds).toEqual([
      { service: "fast", tokensPerSecond: 20, effective: false },
      { service: "plain", tokensPerSecond: 15, effective: true },
      { service: "slow", tokensPerSecond: 10, effective: false },
    ]);
  });
});
