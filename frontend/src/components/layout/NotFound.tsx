// The view for a path no route matches.

import { useTranslation } from "react-i18next";

import { EmptyState } from "../ui/feedback/EmptyState.tsx";
import { PageHeader } from "../ui/layout/PageHeader.tsx";

export function NotFound() {
  const { t } = useTranslation();
  return (
    <div className="flex h-full min-h-0 flex-col">
      <PageHeader title={t("notFound.title")} />
      <div className="min-h-0 flex-1 overflow-auto">
        <EmptyState message={t("notFound.message")} />
      </div>
    </div>
  );
}
