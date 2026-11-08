// The draft's validation errors under the editor, each with the line and
// column it points at.

import { useTranslation } from "react-i18next";

import type { ValidationError } from "../../api/client.ts";

export function ValidationPanel({
  errors,
}: {
  readonly errors: readonly ValidationError[];
}) {
  const { t } = useTranslation();
  return (
    <section
      aria-label={t("config.validationErrors")}
      className="max-h-40 shrink-0 overflow-auto border-t border-danger-line bg-surface px-4 py-2"
    >
      <h2 className="mb-1 text-xs font-semibold text-danger">
        {t("config.validationErrors")}
      </h2>
      <ul className="flex flex-col gap-0.5">
        {errors.map((e, i) => (
          <li key={i} className="font-mono text-xs wrap-anywhere text-danger">
            <span className="text-muted">
              {t("config.lineColumn", { line: e.line, column: e.column })}
            </span>{" "}
            {e.message}
          </li>
        ))}
      </ul>
    </section>
  );
}
