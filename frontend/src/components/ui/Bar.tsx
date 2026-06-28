// The allocation rail — the console's signature element. A memory
// budget is fixed, so this reads like an instrument gauge: segments
// fill a recessed track marked with quarter ticks, and the `growth`
// segment (pledged-but-not-yet-resident headroom) is the one place the
// brass accent appears, spotlighting the constraint the daemon plans
// around.

export type BarSegment = {
  variant: "used" | "reserved" | "headroom" | "growth";
  bytes: number;
  label?: string;
};

type BarProps = {
  total: number;
  segments: BarSegment[];
  className?: string;
};

const SEGMENT_CLASSES: Record<BarSegment["variant"], string> = {
  used: "bg-accent",
  reserved: "bg-success",
  growth: "bg-brass",
  headroom: "bg-transparent",
};

// Quarter ticks drawn over the recessed track so a glance reads
// fill proportion without a numeric scale. A 1px rule at every 25%.
const TICKS =
  "bg-[repeating-linear-gradient(90deg,var(--color-border-strong)_0_1px,transparent_1px_25%)]";

export function Bar({ total, segments, className = "" }: BarProps) {
  return (
    <div
      className={`relative h-2.5 overflow-hidden rounded-[3px] bg-base ring-1 ring-inset ring-border-default ${className}`}
    >
      {total > 0 && (
        <div className="flex h-full">
          {segments.map((seg, i) => {
            const pct = (seg.bytes / total) * 100;
            if (pct <= 0) return null;
            return (
              <div
                key={i}
                className={`${SEGMENT_CLASSES[seg.variant]} transition-all`}
                style={{ width: `${pct}%` }}
                title={seg.label}
              />
            );
          })}
        </div>
      )}
      <div
        className={`pointer-events-none absolute inset-0 opacity-50 ${TICKS}`}
      />
    </div>
  );
}
