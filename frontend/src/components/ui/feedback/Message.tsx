// One line of inline status: loading, empty, or an error. Errors are
// announced assertively; the others politely.

import { useTranslation } from "react-i18next";

import { cn } from "../cn.ts";

export type MessageProps = (
  | { readonly kind: "loading" }
  | { readonly kind: "empty"; readonly text: string }
  | { readonly kind: "error"; readonly text: string }
) & { readonly className?: string };

export function Message(props: MessageProps) {
  const { t } = useTranslation();
  if (props.kind === "error") {
    return (
      <p
        role="alert"
        className={cn(
          "py-2 text-sm wrap-anywhere text-danger",
          props.className,
        )}
      >
        {props.text}
      </p>
    );
  }
  return (
    <p role="status" className={cn("py-2 text-sm text-muted", props.className)}>
      {props.kind === "loading" ? t("common.loading") : props.text}
    </p>
  );
}
