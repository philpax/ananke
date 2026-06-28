// Engraved label + mono value. Used for quick stats and inline data
// readouts. The label uses the console's eyebrow treatment so stat
// clusters read like a gauge legend.

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
      <div className="eyebrow">{label}</div>
      <div className="mt-0.5 font-mono text-sm text-primary">
        {value}
        {unit && <span className="ml-0.5 text-tertiary">{unit}</span>}
      </div>
    </div>
  );
}
