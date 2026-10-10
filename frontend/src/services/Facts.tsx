import type { Detail, Service } from "../api/contract";
import { memory } from "../ui/format";

export function Facts({
  detail,
  summary,
}: {
  readonly detail: Detail;
  readonly summary?: Service;
}) {
  const estimate = detail.estimate;
  const estimateRows = estimate
    ? [
        ["Weights", memory(estimate.weights_bytes)],
        [
          `KV cache at ${estimate.configured_context.toLocaleString()} ctx`,
          memory(estimate.kv_bytes_for_context),
        ],
        [
          "Compute per device",
          memory(estimate.compute_buffer_bytes_per_device),
        ],
        ["Total", memory(summary?.footprint_bytes)],
      ]
    : [];
  const configRows = [
    ["Port", `:${detail.port}`],
    ["Footprint", memory(summary?.footprint_bytes)],
    [
      "Lifecycle",
      detail.lifecycle === "persistent" ? "Persistent" : "On demand",
    ],
    ["Priority", String(detail.priority)],
  ];
  return (
    <>
      <section class="min-w-0">
        <h2 class="pb-[9px] text-[13px] leading-[1.35] font-semibold">
          Memory estimate
        </h2>
        {estimate ? (
          <div>
            {estimateRows.map(([label, value]) => (
              <div
                key={label}
                class="
                  flex items-baseline gap-3 border-b border-line-subtle py-[9px]
                  text-[13px]
                  last:border-0
                "
              >
                <span class="text-body">{label}</span>
                <span class="ml-auto tabular-nums">{value}</span>
              </div>
            ))}
          </div>
        ) : (
          <p class="py-2 text-[13px] text-muted">
            The backend has no model memory estimate for this service.
          </p>
        )}
      </section>
      <section class="min-w-0" aria-label="Service configuration">
        <h2 class="mb-3 text-[13px] leading-[1.35] font-semibold">
          Configuration
        </h2>
        <table class="w-full border-collapse text-[13px]">
          <tbody>
            {configRows.map(([label, value]) => (
              <tr key={label} class="border-b border-line-subtle">
                <th class="py-2 text-left font-normal text-muted">{label}</th>
                <td class="py-2 text-right">{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
