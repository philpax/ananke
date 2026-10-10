import type { ComponentChildren } from "preact";

export function PageHeader({
  title,
  meta,
  children,
}: {
  readonly title: string;
  readonly meta?: ComponentChildren;
  readonly children?: ComponentChildren;
}) {
  return (
    <header
      class="
        flex min-h-14 shrink-0 flex-wrap items-center gap-3 border-b border-line
        bg-surface px-4 py-2.5
      "
    >
      <div class="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1.5">
        <h1 class="m-0 text-base leading-[1.3] font-semibold">{title}</h1>
        <span class="text-[12.5px] leading-[1.4] wrap-anywhere text-muted">
          {meta}
        </span>
      </div>
      {children && (
        <div class="ml-auto flex flex-wrap items-center gap-2">{children}</div>
      )}
    </header>
  );
}
