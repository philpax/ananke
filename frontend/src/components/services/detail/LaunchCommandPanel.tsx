// The command the service would launch with, standalone and under the
// current placement conditions. Computing it renders the placement, so it
// is fetched only once the operator expands it.

import { useState } from "react";
import { useTranslation } from "react-i18next";

import type { LaunchCommand } from "../../../api/client.ts";
import { useServiceCommand } from "../../../api/hooks.ts";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { Message } from "../../ui/feedback/Message.tsx";
import { Panel } from "../../ui/layout/Panel.tsx";
import { Chip } from "../../ui/status/Chip.tsx";
import { renderCommand } from "./launchCommandFormat.ts";

export function LaunchCommandPanel({ name }: { readonly name: string }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const { data, error, isPending } = useServiceCommand(name, open);
  return (
    <Panel heading={t("serviceDetail.launchCommand")}>
      <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary className="cursor-pointer text-sm text-soft select-none hover:text-ink">
          {t("serviceDetail.expandToCompute")}
        </summary>
        <div className="mt-3">
          {open && isPending && <Message kind="loading" />}
          {error && (
            <Message
              kind="error"
              text={t("serviceDetail.cannotCompute", { error: error.message })}
            />
          )}
          {data && (
            <div className="grid grid-cols-1 gap-3 @min-detail-2col/detail:grid-cols-2">
              <CommandPanel
                label={t("serviceDetail.standalone")}
                command={data.on_empty}
              />
              <CommandPanel
                label={t("serviceDetail.currentConditions")}
                command={data.active ?? null}
              />
            </div>
          )}
        </div>
      </details>
    </Panel>
  );
}

export function CommandPanel({
  label,
  command,
}: {
  readonly label: string;
  readonly command: LaunchCommand | null;
}) {
  const { t } = useTranslation();
  const rendered = command && renderCommand(command);
  return (
    <div className="min-w-0">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <h3 className="text-xs text-muted">{label}</h3>
        {command?.container ? (
          <Chip tone="accent">
            {command.container.runtime} · {command.container.image}
          </Chip>
        ) : (
          command && (
            <span
              title={t(
                command.env_inherit
                  ? "serviceDetail.envInheritOnHint"
                  : "serviceDetail.envInheritOffHint",
              )}
            >
              <Chip tone={command.env_inherit ? "neutral" : "warn"}>
                {t(
                  command.env_inherit
                    ? "serviceDetail.envInheritOn"
                    : "serviceDetail.envInheritOff",
                )}
              </Chip>
            </span>
          )
        )}
        {rendered && <CopyButton value={rendered} label={label} />}
      </div>
      {rendered ? (
        <pre className="overflow-x-auto border border-line bg-canvas p-2 font-mono text-xs text-ink">
          {rendered}
        </pre>
      ) : (
        <p className="border border-line bg-canvas p-4 text-center text-xs text-danger">
          {t("serviceDetail.doesNotFitCurrent")}
        </p>
      )}
    </div>
  );
}
