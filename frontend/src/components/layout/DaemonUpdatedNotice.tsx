// A non-blocking notice shown when the live socket reconnects to a daemon
// with a different build than this tab loaded: the page's code may no longer
// match the API, so offer a reload rather than forcing one.

import { useTranslation } from "react-i18next";

import { useDaemonUpdated } from "../../api/live/useLive.ts";
import { Button } from "../ui/button/Button.tsx";

export function DaemonUpdatedNotice() {
  const { t } = useTranslation();
  if (!useDaemonUpdated()) return null;
  return (
    <div
      role="status"
      className="flex items-center justify-center gap-3 border-b border-line bg-raised px-4 py-1.5 text-xs text-body"
    >
      <span>{t("app.daemonUpdated")}</span>
      <Button size="sm" onClick={() => window.location.reload()}>
        {t("app.reload")}
      </Button>
    </div>
  );
}
