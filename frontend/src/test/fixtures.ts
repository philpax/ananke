// Complete wire payloads for component tests, so a test states only the
// fields it exercises and never casts a partial one into shape.

import type { ServiceDetail } from "../api/client.ts";

export function serviceDetail(
  overrides: Partial<ServiceDetail> = {},
): ServiceDetail {
  return {
    name: "muse-glimmer",
    state: "running",
    lifecycle: "on_demand",
    priority: 50,
    port: 8000,
    private_port: 48202,
    template: "llama-cpp",
    openai_compat: true,
    placement: {
      policy: "gpu-only",
      gpu_headroom_bytes: 0,
      split: { mode: "layer" },
      allocation: { kind: "estimated" },
    },
    placement_override: {},
    idle_timeout_ms: 600_000,
    rolling_samples: 0,
    rolling_samples_host: 0,
    observed_peak_bytes: 0,
    current_allocation: {},
    ...overrides,
  };
}
