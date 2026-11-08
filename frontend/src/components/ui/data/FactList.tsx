// A list of labelled facts: muted labels beside their values, the `dl`
// grid every detail panel repeats.

import type { ReactNode } from "react";

import { cn } from "../cn.ts";

export type Fact = {
  readonly label: ReactNode;
  readonly value: ReactNode;
  /// Monospace, for ids, paths, sizes, and other machine values.
  readonly mono?: boolean;
  readonly title?: string;
};

export function FactList({
  facts,
  className,
}: {
  /// Falsy entries are skipped, so `cond && { … }` lists an optional fact.
  readonly facts: readonly (Fact | null | undefined | false | "" | 0)[];
  readonly className?: string;
}) {
  return (
    <dl
      className={cn(
        "grid grid-cols-[minmax(7rem,auto)_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm",
        className,
      )}
    >
      {facts.map((fact, index) =>
        fact ? (
          // Index keys: a fact list is static per render and labels may
          // be elements.
          <div key={index} className="contents">
            <dt className="text-muted">{fact.label}</dt>
            <dd
              title={fact.title}
              className={cn(
                "min-w-0 wrap-anywhere text-ink",
                fact.mono && "font-mono",
              )}
            >
              {fact.value}
            </dd>
          </div>
        ) : null,
      )}
    </dl>
  );
}
