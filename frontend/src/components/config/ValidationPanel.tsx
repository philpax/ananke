import { useTranslation } from "react-i18next";

import type { ValidationError } from "../../api/client.ts";
import { formatValidationError } from "./validation.ts";

export function ValidationPanel({ errors }: { errors: ValidationError[] }) {
  const { t } = useTranslation();
  return (
    <div className="max-h-40 shrink-0 overflow-auto border-t border-border-default bg-surface px-4 py-2">
      <div className="eyebrow mb-1 text-danger">
        {t("config.validationErrors")}
      </div>
      <ul className="space-y-0.5">
        {errors.map((error, index) => (
          <li
            key={`${error.code}-${index}`}
            className="font-mono text-xs text-danger"
          >
            {formatValidationError(error)}
          </li>
        ))}
      </ul>
    </div>
  );
}
