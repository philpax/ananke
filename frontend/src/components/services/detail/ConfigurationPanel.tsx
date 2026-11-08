// How the service is set up: its template, lifecycle, API exposure,
// container identity, and a command service's settings that its launch
// command does not show.

import { useTranslation } from "react-i18next";

import type { ServiceDetail } from "../../../api/client.ts";
import {
  formatDuration,
  formatTimestamp,
  relativeTime,
} from "../../../util.ts";
import { FactList } from "../../ui/data/FactList.tsx";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { Panel } from "../../ui/layout/Panel.tsx";

export function ConfigurationPanel({
  detail,
}: {
  readonly detail: ServiceDetail;
}) {
  const { t } = useTranslation();
  const { container, command } = detail;
  const shutdown = command?.shutdown_command?.join(" ");
  return (
    <Panel heading={t("serviceDetail.configuration")}>
      <FactList
        facts={[
          {
            label: t("serviceDetail.template"),
            value: detail.template,
            mono: true,
          },
          {
            label: t("serviceDetail.lifecycle"),
            value: t(`serviceDetail.lifecycles.${detail.lifecycle}`),
          },
          {
            label: t("serviceDetail.priority"),
            value: detail.priority,
            mono: true,
          },
          {
            label: t("serviceDetail.openaiApi"),
            value: detail.openai_compat
              ? t("serviceDetail.openaiListed")
              : t("serviceDetail.openaiNotListed"),
          },
          command?.upstream_model && {
            label: t("serviceDetail.upstreamModel"),
            value: command.upstream_model,
            mono: true,
          },
          container && {
            label: t("serviceDetail.containerRuntime"),
            value: `${container.runtime} · ${container.image}`,
            mono: true,
          },
          container && {
            label: t("serviceDetail.containerNetwork"),
            value: container.network,
            mono: true,
          },
          container?.container_id && {
            label: t("serviceDetail.containerId"),
            value: (
              <span className="inline-flex items-center gap-1">
                <span className="font-mono text-xs">
                  {container.container_id.slice(0, 12)}
                </span>
                <CopyButton value={container.container_id} />
              </span>
            ),
          },
          container?.container_name && {
            label: t("serviceDetail.containerName"),
            value: container.container_name,
            mono: true,
          },
          {
            label: t("serviceDetail.privatePort"),
            value:
              command === undefined || command === null
                ? `:${detail.private_port}`
                : t(
                    command.private_port_fixed
                      ? "serviceDetail.portFixed"
                      : "serviceDetail.portFromPool",
                    { port: detail.private_port },
                  ),
            mono: true,
          },
          command?.workdir && {
            label: t("serviceDetail.workdir"),
            value: command.workdir,
            mono: true,
          },
          shutdown && {
            label: t("serviceDetail.shutdownCommand"),
            value: (
              <span className="inline-flex items-start gap-1">
                <span className="font-mono text-xs break-all">{shutdown}</span>
                <CopyButton
                  value={shutdown}
                  label={t("serviceDetail.shutdownCommand")}
                />
              </span>
            ),
          },
          {
            label: t("serviceDetail.idleTimeout"),
            value:
              detail.lifecycle === "persistent"
                ? t("serviceDetail.neverPersistent")
                : formatDuration(detail.idle_timeout_ms),
            mono: true,
          },
          {
            label: t("serviceDetail.lastUsed"),
            value:
              detail.last_used_ms != null
                ? relativeTime(detail.last_used_ms)
                : "—",
            title:
              detail.last_used_ms != null
                ? formatTimestamp(detail.last_used_ms)
                : undefined,
            mono: true,
          },
          {
            label: t("serviceDetail.runId"),
            value: detail.run_id ?? "—",
            mono: true,
          },
        ]}
      />
    </Panel>
  );
}
