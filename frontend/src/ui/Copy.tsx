import clsx from "clsx";
import { useEffect, useRef, useState } from "preact/hooks";

import { Icon } from "./Icon";

export function Copy({
  text,
  label = text,
}: {
  readonly text: string;
  readonly label?: string;
}) {
  const [status, setStatus] = useState("idle");
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timerRef.current), []);
  const name =
    status === "copied"
      ? `Copied ${label} to clipboard`
      : status === "failed"
        ? `Copy failed: ${label}`
        : `Copy ${label} to clipboard`;
  return (
    <button
      type="button"
      title={name}
      aria-label={name}
      class={clsx(
        `
          inline-flex size-6 shrink-0 items-center justify-center border-0
          bg-transparent p-0 leading-none
          hover:text-ink
        `,
        status === "copied" ? `text-ok` : `text-soft`,
      )}
      onKeyDown={(e) => e.stopPropagation()}
      onClick={async (e) => {
        e.stopPropagation();
        try {
          await navigator.clipboard.writeText(text);
          setStatus("copied");
        } catch {
          setStatus("failed");
        }
        clearTimeout(timerRef.current);
        timerRef.current = setTimeout(() => setStatus("idle"), 3000);
      }}
    >
      <Icon name={status === "copied" ? "check" : "copy"} size={13} />
      <span role="status" class="sr-only">
        {status === "failed"
          ? "Clipboard is unavailable."
          : status === "copied"
            ? "Copied."
            : ""}
      </span>
    </button>
  );
}
