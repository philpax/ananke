// Horizontal proportion bar for VRAM/placement visualisation.
// Shows used-by-others vs. this-service vs. growth headroom.

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
  headroom: "bg-elevated",
  growth: "bg-warning",
};

export function Bar({ total, segments, className = "" }: BarProps) {
  if (total <= 0) {
    return <div className={`h-2 rounded-sm bg-elevated ${className}`} />;
  }
  return (
    <div
      className={`flex h-2 overflow-hidden rounded-sm bg-elevated ${className}`}
    >
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
  );
}
