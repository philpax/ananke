// The band's left end: the mark and wordmark, the address this tab
// reached, the connection state, and the theme toggle. On narrow screens
// it is the band's single row; `summary` slots in before the toggle there.

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { useDaemonBuild } from "../../../api/live/useLive.ts";
import { cn } from "../../ui/cn.ts";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { Icon } from "../../ui/icon/Icon.tsx";
import { ThemeToggle } from "../../ui/theme/ThemeToggle.tsx";
import { ConnectionStatus } from "../ConnectionStatus.tsx";

type BandIdentityProps = {
  /// Shown beside the connection state on wide screens.
  readonly uptime?: ReactNode;
  /// The narrow-screen resource summary; hidden from `md` up.
  readonly summary?: ReactNode;
};

export function BandIdentity({ uptime, summary }: BandIdentityProps) {
  const { t } = useTranslation();
  const build = useDaemonBuild();
  const host = window.location.host;
  return (
    <div className="flex min-h-touch min-w-0 items-center gap-2 px-2.5 py-1 md:w-band-identity md:shrink-0 md:gap-3 md:bg-selected-soft/20 md:px-3.5 md:py-2">
      <span
        className="flex shrink-0 items-center gap-2 text-accent"
        title={build === null ? undefined : t("shell.build", { build })}
      >
        <Icon name="mark" className="size-5 md:size-6.5" />
        <span className="font-mono text-lg leading-none font-bold max-[22.5rem]:sr-only md:text-xl">
          ananke
        </span>
      </span>
      <div className="flex min-w-0 flex-row-reverse items-center gap-2 md:ml-auto md:flex-col md:items-end md:gap-1">
        <span className="inline-flex min-w-0 items-center gap-0.5 text-body max-sm:hidden">
          <span className="truncate font-mono text-xs leading-none">
            {host}
          </span>
          <CopyButton value={host} />
        </span>
        <span className="flex items-center gap-2">
          <ConnectionStatus />
          {uptime !== undefined && (
            <span className="text-xs whitespace-nowrap text-muted tabular-nums max-md:hidden">
              {uptime}
            </span>
          )}
        </span>
      </div>
      {summary !== undefined && (
        <div className="ml-auto flex shrink-0 md:hidden">{summary}</div>
      )}
      <ThemeToggle
        className={cn(
          "max-md:size-touch",
          summary === undefined && "max-md:ml-auto",
        )}
      />
    </div>
  );
}
