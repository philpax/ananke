import type { ValidationError } from "../../api/client.ts";

/** Format one diagnostic for visible config-editor output. */
export function formatValidationError(error: ValidationError): string {
  const location = error.location
    ? `L${error.location.line}:${error.location.column}`
    : "";
  const path = error.path ? ` ${error.path}` : "";
  return [location, `[${error.code}]${path}`, error.message]
    .filter((part) => part !== "")
    .join(" ");
}
