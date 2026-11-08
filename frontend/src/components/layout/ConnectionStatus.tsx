// The live socket's state as a dot and, on wide screens, a word. While the
// socket is down the band and views show the last snapshot they had, so
// the reconnecting state says that rather than only that it is retrying.

import { useTranslation } from "react-i18next";

import type { LiveStatus } from "../../api/live/socket.ts";
import { useLiveStatus } from "../../api/live/useLive.ts";
import { cn } from "../ui/cn.ts";
import { StatusDot } from "../ui/status/StatusDot.tsx";
import { toneText, type Tone } from "../ui/tone.ts";

export function ConnectionStatus({
  className,
}: {
  readonly className?: string;
}) {
  const { t } = useTranslation();
  const status = useLiveStatus();
  const { tone, short, full } = STATUS_VIEW[status];
  return (
    <span
      role="status"
      className={cn(
        "inline-flex items-center gap-1.5 text-xs whitespace-nowrap",
        toneText({ tone }),
        className,
      )}
    >
      <StatusDot tone={tone} />
      {short !== undefined && (
        <span aria-hidden="true" className="max-md:hidden">
          {t(short)}
        </span>
      )}
      <span className="sr-only">{t(full)}</span>
    </span>
  );
}

const STATUS_VIEW: Readonly<
  Record<
    LiveStatus,
    { readonly tone: Tone; readonly short?: string; readonly full: string }
  >
> = {
  open: { tone: "ok", full: "shell.connection.open" },
  connecting: {
    tone: "neutral",
    short: "shell.connection.connecting",
    full: "shell.connection.connecting",
  },
  reconnecting: {
    tone: "warn",
    short: "shell.connection.reconnecting",
    full: "shell.connection.reconnectingDetail",
  },
};
