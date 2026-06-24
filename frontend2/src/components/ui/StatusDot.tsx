// Small coloured dot for service state. Quieter than a badge — only
// non-idle states get a visible dot. Idle/stopped/disabled services
// get a faint grey dot so the eye skips them.

type StatusDotProps = {
  state: string;
  className?: string;
};

function dotColour(state: string): string {
  if (state === "running") return "bg-success";
  if (state === "starting") return "bg-warning";
  if (state === "draining") return "bg-warning";
  if (state === "failed") return "bg-danger";
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
