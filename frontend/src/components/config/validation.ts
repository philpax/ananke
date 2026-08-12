import type { ValidationError } from "../../api/client.ts";

/** Format one diagnostic for visible config-editor output. */
export function formatValidationError(error: ValidationError): string {
  const location = error.location
    ? `L${error.location.line}:${error.location.column}`
    : "";
  const path = error.path ? ` ${error.path}` : "";
  const context = formatContext(error.context);
  return [location, `[${error.code}]${path}`, context, error.message]
    .filter((part) => part !== "")
    .join(" ");
}

function formatContext(context: ValidationError["context"]): string {
  switch (context.kind) {
    case "Service":
      return context.data.service
        ? `service=${context.data.service}`
        : context.data.index !== null && context.data.index !== undefined
          ? `service[${context.data.index}]`
          : "";
    case "Field":
      return context.data.offending
        ? `${context.data.field}=${context.data.offending}`
        : context.data.field;
    case "Value":
      return `${context.data.field}=${context.data.offending}`;
    case "Count":
      return `${context.data.field}=${context.data.got}/${context.data.expected}`;
    case "Index":
      return `${context.data.field}[${context.data.index}]=${context.data.value}`;
    case "Fields":
      return context.data.fields.join(", ");
    case "Placeholder":
      return context.data.argument
        ? `${context.data.field}=${context.data.argument}`
        : context.data.field;
    case "Merge":
      return context.data.parent
        ? `parent=${context.data.parent}`
        : context.data.service
          ? `service=${context.data.service}`
          : "";
    case "Parse":
      return "";
    case "Other":
      return "";
    default: {
      const _: never = context;
      return _;
    }
  }
}
