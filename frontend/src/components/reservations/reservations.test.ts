import { describe, expect, it } from "vitest";

import type { DeviceSummary } from "../../api/client.ts";
import {
  EMPTY_FORM,
  deviceBudgets,
  deviceEntries,
  formRequest,
} from "./reservations.ts";

const GIB = 1024 ** 3;

const devices: DeviceSummary[] = [
  {
    id: "gpu:0",
    name: "RTX 3090",
    total_bytes: 24 * GIB,
    free_bytes: 10 * GIB,
    reservations: [
      { service: "qwen", bytes: 6 * GIB, elastic: false },
      {
        service: "reservation:3",
        reservation: 3,
        bytes: 8 * GIB,
        elastic: false,
      },
    ],
  },
  {
    id: "cpu",
    name: "CPU",
    total_bytes: 64 * GIB,
    free_bytes: 60 * GIB,
    reservations: [],
  },
];

describe("deviceBudgets", () => {
  it("splits manual reservations from service pledges", () => {
    expect(deviceBudgets(devices)).toEqual([
      {
        id: "gpu:0",
        name: "RTX 3090",
        total: 24 * GIB,
        manual: 8 * GIB,
        services: 6 * GIB,
        unpledged: 10 * GIB,
      },
      {
        id: "cpu",
        name: "CPU",
        total: 64 * GIB,
        manual: 0,
        services: 0,
        unpledged: 64 * GIB,
      },
    ]);
  });
});

describe("formRequest", () => {
  const budgets = deviceBudgets(devices);

  it("builds a request from the filled amounts", () => {
    expect(
      formRequest(
        {
          label: " train ",
          amounts: { "gpu:0": "1.5", cpu: "" },
        },
        budgets,
      ),
    ).toEqual({
      ok: true,
      request: {
        devices: { "gpu:0": 1.5 * GIB },
        label: "train",
      },
    });
  });

  it("leaves the label out when blank", () => {
    expect(
      formRequest({ ...EMPTY_FORM, amounts: { cpu: "2" } }, budgets),
    ).toEqual({ ok: true, request: { devices: { cpu: 2 * GIB } } });
  });

  it.each([
    [{ ...EMPTY_FORM }, { kind: "no_devices" }],
    [
      { ...EMPTY_FORM, amounts: { cpu: "x" } },
      { kind: "bad_amount", device: "cpu" },
    ],
    [
      { ...EMPTY_FORM, amounts: { cpu: "0" } },
      { kind: "bad_amount", device: "cpu" },
    ],
    [
      { ...EMPTY_FORM, amounts: { "gpu:0": "25" } },
      { kind: "too_large", device: "gpu:0" },
    ],
  ])("rejects %j", (form, problem) => {
    expect(formRequest(form, budgets)).toEqual({ ok: false, problem });
  });
});

describe("deviceEntries", () => {
  it("orders devices by id", () => {
    expect(deviceEntries({ "gpu:1": 2, cpu: 1, "gpu:0": 3 })).toEqual([
      ["cpu", 1],
      ["gpu:0", 3],
      ["gpu:1", 2],
    ]);
  });
});
