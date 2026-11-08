// The bars between the header and the editor: why the config cannot be
// edited, that the file changed on disk under unsaved edits, and why the
// last save failed.

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "../ui/button/Button.tsx";
import { cn } from "../ui/cn.ts";
import { toneBox, type Tone } from "../ui/tone.ts";

export function ReadOnlyNotice() {
  const { t } = useTranslation();
  return <Notice tone="neutral">{t("config.readOnlyNotice")}</Notice>;
}

/// The file changed on disk while the draft had edits. Neither choice is
/// made for the operator: the draft stays until they pick one.
export function DiskChangeNotice({
  onReload,
  onKeep,
}: {
  readonly onReload: () => void;
  readonly onKeep: () => void;
}) {
  const { t } = useTranslation();
  return (
    <Notice
      tone="warn"
      role="alert"
      actions={
        <>
          <Button size="sm" onClick={onKeep}>
            {t("config.keepLocalChanges")}
          </Button>
          <Button size="sm" tone="primary" onClick={onReload}>
            {t("config.reloadFromServer")}
          </Button>
        </>
      }
    >
      {t("config.changedOnDiskNotice")}
    </Notice>
  );
}

export function SaveErrorNotice({ message }: { readonly message: string }) {
  const { t } = useTranslation();
  return (
    <Notice tone="danger" role="alert">
      {t("config.saveFailed", { message })}
    </Notice>
  );
}

function Notice({
  tone,
  role,
  actions,
  children,
}: {
  readonly tone: Tone;
  readonly role?: "alert";
  readonly actions?: ReactNode;
  readonly children: ReactNode;
}) {
  return (
    <div
      role={role}
      className={cn(
        "flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-x-0 border-t-0 px-4 py-2 text-sm",
        toneBox({ tone }),
      )}
    >
      <p className="min-w-0 flex-1 basis-60 wrap-anywhere">{children}</p>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}
