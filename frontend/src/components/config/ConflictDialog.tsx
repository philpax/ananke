// A save was refused because the file changed since the draft was read
// (HTTP 412). The operator keeps their draft, which the next save then
// writes over the new version, or reloads the version on disk. Escape
// keeps the draft, the choice that loses nothing.

import { useEffect, useId, useRef } from "react";
import { Trans, useTranslation } from "react-i18next";

import { Button } from "../ui/button/Button.tsx";

export function ConflictDialog({
  serverHash,
  onKeep,
  onReload,
}: {
  readonly serverHash: string;
  readonly onKeep: () => void;
  readonly onReload: () => void;
}) {
  const { t } = useTranslation();
  const titleId = useId();
  const bodyId = useId();
  const keepRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    keepRef.current?.focus();
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-canvas/80 p-4">
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
        className="w-full max-w-md border border-line-strong bg-surface"
        onKeyDown={(e) => {
          if (e.key === "Escape") onKeep();
        }}
      >
        <h2
          id={titleId}
          className="border-b border-line px-4 py-2.5 text-sm font-semibold text-ink"
        >
          {t("config.changedOnDisk")}
        </h2>
        <p id={bodyId} className="px-4 py-3 text-sm text-body">
          <Trans
            i18nKey="config.changedOnDiskBody"
            components={[<code className="font-mono text-xs text-ink" />]}
            values={{ hash: serverHash.slice(0, 12) }}
          />
        </p>
        <div className="flex flex-wrap justify-end gap-2 px-4 pb-4">
          <Button ref={keepRef} onClick={onKeep}>
            {t("config.keepLocalChanges")}
          </Button>
          <Button tone="primary" onClick={onReload}>
            {t("config.reloadFromServer")}
          </Button>
        </div>
      </div>
    </div>
  );
}
