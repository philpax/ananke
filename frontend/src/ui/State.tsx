import clsx from "clsx";

import { stateLabel, stateTone } from "./format";

export function State({ state }: { readonly state: string }) {
  const tone = stateTone(state);
  return (
    <span
      class={clsx(
        `
          inline-flex items-center gap-[7px] border px-2 py-[3px] text-[12.5px]
          font-semibold
        `,
        {
          "border-ok-line bg-ok-bg text-ok": tone === "ok",
          "border-danger-line bg-danger-bg text-danger": tone === "danger",
          "border-warn-line bg-warn-bg text-warn": tone === "warn",
          "border-line bg-surface text-body": tone === "neutral",
        },
      )}
    >
      <span
        aria-hidden="true"
        class="size-[7px] shrink-0 rounded-full bg-current"
      />
      {stateLabel(state)}
    </span>
  );
}
