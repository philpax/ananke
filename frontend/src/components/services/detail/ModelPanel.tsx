// Facts read from the service's GGUF: what the model is and how big.

import { useTranslation } from "react-i18next";

import type { ModelInfo } from "../../../api/client.ts";
import { formatBytes, formatParameterCount } from "../../../util.ts";
import { FactList } from "../../ui/data/FactList.tsx";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { Panel } from "../../ui/layout/Panel.tsx";

export function ModelPanel({ model }: { readonly model: ModelInfo }) {
  const { t } = useTranslation();
  const tokens = (value: number) =>
    t("serviceDetail.tokensValue", { value: value.toLocaleString() });
  return (
    <Panel heading={t("serviceDetail.model")}>
      <FactList
        facts={[
          model.model_name && {
            label: t("serviceDetail.name"),
            value: model.model_name,
          },
          {
            label: t("serviceDetail.file"),
            value: (
              <span className="inline-flex items-center gap-1">
                <span className="font-mono text-xs break-all">
                  {model.file_name}
                </span>
                <CopyButton value={model.file_name} />
              </span>
            ),
          },
          {
            label: t("serviceDetail.architecture"),
            value: model.architecture,
            mono: true,
          },
          model.parameter_count != null && {
            label: t("serviceDetail.parameters"),
            value: formatParameterCount(model.parameter_count),
            title: t("serviceDetail.parametersExact", {
              value: model.parameter_count.toLocaleString(),
            }),
            mono: true,
          },
          {
            label: t("serviceDetail.onDisk"),
            value: formatBytes(model.total_tensor_bytes),
            mono: true,
          },
          model.block_count != null && {
            label: t("serviceDetail.layers"),
            value: model.block_count,
            mono: true,
          },
          model.trained_context_length != null && {
            label: t("serviceDetail.trainedContext"),
            value: tokens(model.trained_context_length),
            mono: true,
          },
          model.shard_count > 1 && {
            label: t("serviceDetail.shards"),
            value: model.shard_count,
            mono: true,
          },
          model.license && {
            label: t("serviceDetail.license"),
            value: model.license,
            mono: true,
          },
        ]}
      />
    </Panel>
  );
}
