// Segmented toggle — a row of mutually-exclusive pill buttons. Used
// for time range selectors, sort order, filter toggles, etc. across
// the dashboard, stats, service detail, events, and logs views.

type Option<T> = {
  label: string;
  value: T;
};

export function SegmentedToggle<T>({
  options,
  selected,
  onChange,
  className = "",
}: {
  options: Option<T>[];
  selected: T;
  onChange: (value: T) => void;
  className?: string;
}) {
  return (
    <div className={`flex items-center gap-1 ${className}`}>
      {options.map((opt) => (
        <button
          key={opt.label}
          onClick={() => onChange(opt.value)}
          className={`rounded-sm px-2 py-0.5 text-xs transition-colors ${
            opt.value === selected
              ? "bg-elevated text-primary"
              : "text-tertiary hover:text-secondary"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
