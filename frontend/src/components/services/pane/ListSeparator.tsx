// The handle between the service list and the detail: drag or use the
// arrow keys to resize, Enter to hide the list. A 1px rule inside a 10px
// hit area. The list's rendered width is also capped at half the view in
// CSS, so on a narrow window `aria-valuenow` can exceed what is drawn.

import type { PointerEvent } from "react";
import { useTranslation } from "react-i18next";

import {
  LIST_WIDTH_MAX,
  LIST_WIDTH_MIN,
  clampListWidth,
  stepListWidth,
} from "./listPane.ts";

export function ListSeparator({
  width,
  controls,
  onResize,
  onHide,
}: {
  readonly width: number;
  readonly controls: string;
  readonly onResize: (width: number) => void;
  readonly onHide: () => void;
}) {
  const { t } = useTranslation();

  function startDrag(e: PointerEvent<HTMLDivElement>) {
    const target = e.currentTarget;
    const startX = e.clientX;
    const startWidth = width;
    target.setPointerCapture(e.pointerId);
    const move = (event: globalThis.PointerEvent) =>
      onResize(clampListWidth(startWidth + event.clientX - startX));
    const finish = () => {
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", finish);
      target.removeEventListener("pointercancel", finish);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", finish);
    target.addEventListener("pointercancel", finish);
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-controls={controls}
      aria-label={t("services.list.resize")}
      aria-valuemin={LIST_WIDTH_MIN}
      aria-valuemax={LIST_WIDTH_MAX}
      aria-valuenow={width}
      tabIndex={0}
      title={t("services.list.resizeHint")}
      onPointerDown={startDrag}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          onHide();
          return;
        }
        const next = stepListWidth(width, e.key, e.shiftKey);
        if (next !== null) {
          e.preventDefault();
          onResize(next);
        }
      }}
      className="absolute inset-y-0 -right-1.5 z-20 w-2.5 cursor-col-resize touch-none outline-none after:absolute after:inset-y-0 after:left-1 after:w-px after:bg-transparent hover:after:bg-accent focus-visible:after:w-0.5 focus-visible:after:bg-accent max-lg:hidden"
    />
  );
}
