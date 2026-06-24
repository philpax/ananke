// Label + value + optional unit, monospace value. Used for quick
// stats and inline data displays.

import type { ReactNode } from "react";

type StatProps = {
  label: ReactNode;
  value: ReactNode;
  unit?: string;
  className?: string;
};

export function Stat({ label, value, unit, className = "" }: StatProps) {
  return (
    <div className={className}>
      <div className="text-xs text-tertiary">{label}</div>
      <div className="font-mono text-sm text-primary">
        {value}
        {unit && <span className="ml-0.5 text-tertiary">{unit}</span>}
      </div>
    </div>
  );
}
