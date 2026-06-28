// Small coloured dot for service state. Quieter than a badge — only
// non-idle states get a visible dot. Idle/stopped/disabled services
// get a faint grey dot so the eye skips them.

type StatusDotProps = {
  state: string;
  className?: string;
};

// Active states get a faint coloured halo so a live service reads like
// a lit indicator lamp; quiet states stay flat and recede.
function dotColour(state: string): string {
  if (state === "running") return "bg-success ring-2 ring-success/25";
  if (state === "starting" || state === "draining")
    return "bg-warning ring-2 ring-warning/25";
  if (state === "failed") return "bg-danger ring-2 ring-danger/25";
  return "bg-tertiary/40";
}

export function StatusDot({ state, className = "" }: StatusDotProps) {
  return (
    <span
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${dotColour(state)} ${className}`}
      title={state}
    />
  );
}
