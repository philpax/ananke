// A llama-cpp service's runtime and serving knobs, including values no
// single config key states: the per-slot context and the resolved prompt
// cache cap.

import { useTranslation } from "react-i18next";

import type { RuntimeInfo, ServingConfig } from "../../../api/client.ts";
import { formatBytes } from "../../../util.ts";
import { FactList } from "../../ui/data/FactList.tsx";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { Panel } from "../../ui/layout/Panel.tsx";

export function ServingPanel({
  serving,
  runtime,
}: {
  readonly serving: ServingConfig;
  readonly runtime: RuntimeInfo | null | undefined;
}) {
  const { t } = useTranslation();
  const flag = (on: boolean) =>
    on ? t("serviceDetail.flagOn") : t("serviceDetail.flagOff");
  const tokens = (value: number) =>
    t("serviceDetail.tokensValue", { value: value.toLocaleString() });
  const pair = (a: number | null | undefined, b: number | null | undefined) =>
    `${a ?? "—"} / ${b ?? "—"}`;
  const binaryName = serving.binary.split("/").at(-1) ?? serving.binary;
  const ik = runtime?.ik;
  return (
    <Panel heading={t("serviceDetail.serving")}>
      <FactList
        facts={[
          runtime && {
            label: t("serviceDetail.runtime"),
            value: runtime.kind,
            mono: true,
          },
          {
            label: t("serviceDetail.binary"),
            value: (
              <span className="inline-flex items-center gap-1">
                <span className="font-mono text-xs">{binaryName}</span>
                <CopyButton value={serving.binary} />
              </span>
            ),
            title: serving.binary,
          },
          ik?.mla != null && {
            label: t("serviceDetail.ikMla"),
            value: ik.mla,
            mono: true,
          },
          ik && { label: t("serviceDetail.ikDsa"), value: flag(ik.dsa) },
          ik?.attn_max_batch != null && {
            label: t("serviceDetail.ikAmb"),
            value: ik.attn_max_batch,
            mono: true,
          },
          ik && {
            label: t("serviceDetail.ikRtr"),
            value: flag(ik.runtime_repack),
          },
          serving.context != null && {
            label: t("serviceDetail.context"),
            value: tokens(serving.context),
            mono: true,
          },
          serving.n_gpu_layers != null && {
            label: t("serviceDetail.gpuLayers"),
            value: serving.n_gpu_layers,
            mono: true,
          },
          {
            label: t("serviceDetail.kvCache"),
            value: `${serving.cache_type_k} / ${serving.cache_type_v}${
              serving.flash_attn ? " · fa" : ""
            }`,
            mono: true,
          },
          {
            label: t("serviceDetail.parallelSlots"),
            value: serving.kv_unified
              ? `${serving.parallel} · ${t("serviceDetail.kvUnified")}`
              : serving.parallel,
            mono: true,
          },
          serving.effective_context_per_slot != null && {
            label: t("serviceDetail.perSlotContext"),
            value: tokens(serving.effective_context_per_slot),
            mono: true,
          },
          {
            label: t("serviceDetail.promptCache"),
            value:
              serving.cache_ram_bytes > 0
                ? formatBytes(serving.cache_ram_bytes)
                : t("serviceDetail.flagOff"),
            mono: true,
          },
          serving.spec_type && {
            label: t("serviceDetail.specDecode"),
            value: serving.draft_model
              ? `${serving.spec_type} · ${serving.draft_model}`
              : serving.spec_type,
            mono: true,
          },
          serving.expert_offload !== "off" && {
            label: t("serviceDetail.expertOffloadMode"),
            value: serving.expert_offload,
            mono: true,
          },
          (serving.batch_size != null || serving.ubatch_size != null) && {
            label: t("serviceDetail.batchSizes"),
            value: pair(serving.batch_size, serving.ubatch_size),
            mono: true,
          },
          (serving.threads != null || serving.threads_batch != null) && {
            label: t("serviceDetail.threadsRow"),
            value: pair(serving.threads, serving.threads_batch),
            mono: true,
          },
          serving.numa && {
            label: t("serviceDetail.numaRow"),
            value: serving.numa,
            mono: true,
          },
          {
            label: t("serviceDetail.memoryFlags"),
            value: `${flag(serving.mmap)} / ${flag(serving.mlock)}`,
          },
        ]}
      />
    </Panel>
  );
}
