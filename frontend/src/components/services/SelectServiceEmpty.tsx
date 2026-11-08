// The detail pane before a service is chosen, with a way back to a list
// the operator has hidden.

import { useTranslation } from "react-i18next";

import { Button } from "../ui/button/Button.tsx";
import { EmptyState } from "../ui/feedback/EmptyState.tsx";

export function SelectServiceEmpty({
  listHidden,
  count,
  onBrowse,
}: {
  readonly listHidden: boolean;
  readonly count: number;
  readonly onBrowse: () => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 bg-canvas p-8">
      <EmptyState message={t("services.selectService")} />
      {listHidden && (
        <Button
          icon="browse"
          onClick={onBrowse}
          className="max-lg:hidden"
          data-browse-services=""
        >
          {t("services.detail.browse", { count })}
        </Button>
      )}
    </div>
  );
}
