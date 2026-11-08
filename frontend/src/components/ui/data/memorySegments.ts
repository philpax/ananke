// Turns a memory budget into bar segments. Every width is clamped so the
// segments never sum past the track, whatever the daemon reports: a
// reservation can briefly exceed the card's total while a model loads.

export type MemoryBarInput =
  | {
      /// A device's memory: what is resident, then what is pledged to
      /// services beyond that.
      readonly variant: "device";
      readonly total: number;
      readonly used: number;
      readonly pledged: number;
    }
  | {
      /// One service's placement on a device: other services' memory,
      /// then this service's, then the headroom it may grow into.
      readonly variant: "placement";
      readonly total: number;
      readonly others: number;
      readonly self: number;
      readonly growth?: number;
    };

export type MemorySegmentKind =
  | "used"
  | "pledged"
  | "others"
  | "self"
  | "growth";

export type MemorySegment = {
  readonly kind: MemorySegmentKind;
  readonly bytes: number;
  readonly percent: number;
};

export function memorySegments(
  input: MemoryBarInput,
): readonly MemorySegment[] {
  const parts: readonly (readonly [MemorySegmentKind, number])[] =
    input.variant === "device"
      ? [
          ["used", input.used],
          ["pledged", input.pledged],
        ]
      : [
          ["others", input.others],
          ["self", input.self],
          ["growth", input.growth ?? 0],
        ];
  if (!(input.total > 0)) return [];

  let remaining = 100;
  const segments: MemorySegment[] = [];
  for (const [kind, raw] of parts) {
    const bytes = Number.isFinite(raw) ? Math.max(0, raw) : 0;
    const percent = Math.min(remaining, (bytes / input.total) * 100);
    remaining -= percent;
    if (percent > 0) segments.push({ kind, bytes, percent });
  }
  return segments;
}
