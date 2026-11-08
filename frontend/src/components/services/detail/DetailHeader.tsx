// The service detail's sticky header: name, state and why, description,
// and the lifecycle actions, with any failure shown under them.

import { useTranslation } from "react-i18next";
import { Link, useLocation } from "react-router-dom";

import type {
  DisableReason,
  ServiceStateKind,
  ServiceSummary,
} from "../../../api/client.ts";
import { Button } from "../../ui/button/Button.tsx";
import { ButtonLink } from "../../ui/button/ButtonLink.tsx";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { Message } from "../../ui/feedback/Message.tsx";
import { rangeSearch } from "../../ui/form/timeRange.ts";
import { Icon } from "../../ui/icon/Icon.tsx";
import { Chip } from "../../ui/status/Chip.tsx";
import { StateChip } from "../../ui/status/StateChip.tsx";
import { availableActions, canChat, serviceOpenUrl } from "./serviceActions.ts";
import { useServiceAction } from "./useServiceAction.ts";

/// What the header needs; the detail supplies it once loaded, the list's
/// summary before then, so the state never blanks while the detail loads.
export type HeaderService = Pick<
  ServiceSummary,
  "name" | "port" | "openai_compat" | "modality" | "description" | "pid"
> & {
  readonly state: ServiceStateKind;
  readonly disabled_reason?: DisableReason | null;
  readonly has_mmproj?: boolean | null;
};

export function DetailHeader({
  service,
  titleId,
  listHidden,
  serviceCount,
  onBrowse,
}: {
  readonly service: HeaderService;
  readonly titleId: string;
  readonly listHidden: boolean;
  readonly serviceCount: number;
  readonly onBrowse: () => void;
}) {
  const { t } = useTranslation();
  const { search } = useLocation();
  const action = useServiceAction(service.name);
  const actions = availableActions(service.state);
  const chattable = canChat(service);

  return (
    <header className="sticky top-0 z-10 flex flex-col gap-2 border-b border-line-strong bg-surface px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <Link
          to={`/${rangeSearch(search)}`}
          className="inline-flex items-center gap-1 text-sm text-soft hover:text-ink lg:hidden"
        >
          <Icon name="back" size="3.5" />
          {t("services.detail.allServices")}
        </Link>
        {listHidden && (
          <Button
            tone="quiet"
            size="sm"
            icon="browse"
            className="max-lg:hidden"
            data-browse-services=""
            onClick={onBrowse}
          >
            {t("services.detail.browse", { count: serviceCount })}
          </Button>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <h1
          id={titleId}
          className="min-w-0 font-mono text-lg font-semibold wrap-anywhere text-ink"
        >
          {service.name}
        </h1>
        <CopyButton value={service.name} />
        <StateChip state={service.state} reason={service.disabled_reason} />
        {service.has_mmproj && <Chip tone="vision">vision</Chip>}
        {service.modality === "embedding" && (
          <Chip tone="embedding">embedding</Chip>
        )}
        {service.pid != null && (
          <span className="font-mono text-xs text-muted">
            {t("services.detail.pid", { pid: service.pid })}
          </span>
        )}
      </div>
      {service.description && (
        <p className="truncate text-sm text-soft" title={service.description}>
          {service.description}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <ButtonLink
          href={serviceOpenUrl(window.location, service.port)}
          external
        >
          {t("services.detail.open", { port: service.port })}
        </ButtonLink>
        {chattable ? (
          <ButtonLink
            tone="primary"
            icon="chat"
            to={`/chat?model=${encodeURIComponent(service.name)}`}
          >
            {t("serviceDetail.chat")}
          </ButtonLink>
        ) : (
          <Button icon="chat" disabled title={chatBlockedReason(service, t)}>
            {t("serviceDetail.chat")}
            <span className="sr-only">{chatBlockedReason(service, t)}</span>
          </Button>
        )}
        {actions.restart && (
          <Button
            icon="restart"
            disabled={action.pending}
            onClick={() => action.run("restart")}
          >
            {t("services.actions.restart")}
          </Button>
        )}
        {actions.toggle === "start" && (
          <Button
            tone="start"
            icon="start"
            disabled={action.pending}
            onClick={() => action.run("start")}
          >
            {t("services.actions.start")}
          </Button>
        )}
        {actions.toggle === "stop" && (
          <Button
            tone="stop"
            icon="stop"
            disabled={action.pending}
            onClick={() => action.run("stop")}
          >
            {t("services.actions.stop")}
          </Button>
        )}
        <Button
          icon={actions.power === "enable" ? "power" : "disable"}
          disabled={action.pending}
          onClick={() => action.run(actions.power)}
        >
          {t(`services.actions.${actions.power}`)}
        </Button>
      </div>
      {action.error && (
        <Message kind="error" text={action.error} className="py-0" />
      )}
    </header>
  );
}

function chatBlockedReason(
  service: HeaderService,
  t: (key: string) => string,
): string {
  if (!service.openai_compat) return t("services.detail.chatNotOpenai");
  if (service.modality === "embedding")
    return t("services.detail.chatEmbedding");
  return t("services.detail.chatDisabled");
}
