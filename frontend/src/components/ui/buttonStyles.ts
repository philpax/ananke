// Shared variant/size classes for button-like elements (`Button`,
// `ButtonLink`, and external anchors styled as buttons) so they stay
// visually identical regardless of the underlying element.
//
// Two registers:
//   - Solid colour variants (red, orange, green, cyan, blue, iris,
//     magenta, slate) draw from one perceptual palette family — the
//     `--color-action-*` tokens share a lightness, so a row of them
//     reads as siblings under white ink. They are named by hue, not by
//     meaning: pick whichever colour sits best in a given button group.
//     `slate` is the neutral member, for buttons with no identity of
//     their own; reach for a hue when a button does carry a meaning.
//   - Quiet variants (secondary, ghost) stay low-contrast for toolbars
//     and inline controls that should recede.

export type ButtonVariant =
  | "red"
  | "orange"
  | "green"
  | "cyan"
  | "blue"
  | "iris"
  | "magenta"
  | "slate"
  | "secondary"
  | "ghost";

export type ButtonSize = "sm" | "md";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  red: "bg-action-red text-white hover:brightness-110",
  orange: "bg-action-orange text-white hover:brightness-110",
  green: "bg-action-green text-white hover:brightness-110",
  cyan: "bg-action-cyan text-white hover:brightness-110",
  blue: "bg-action-blue text-white hover:brightness-110",
  iris: "bg-action-iris text-white hover:brightness-110",
  magenta: "bg-action-magenta text-white hover:brightness-110",
  slate: "bg-action-slate text-white hover:brightness-110",
  secondary: "bg-elevated text-primary hover:bg-border-strong",
  ghost: "text-secondary hover:text-primary hover:bg-elevated",
};

const SIZE_CLASSES: Record<ButtonSize, string> = {
  sm: "h-7 px-2 text-xs",
  md: "h-8 px-3 text-sm",
};

const BASE =
  "inline-flex items-center gap-1.5 rounded-md font-medium transition-[filter,background-color,color] disabled:cursor-not-allowed disabled:opacity-50";

export function buttonClassName(
  variant: ButtonVariant = "secondary",
  size: ButtonSize = "md",
  className = "",
): string {
  return `${BASE} ${VARIANT_CLASSES[variant]} ${SIZE_CLASSES[size]} ${className}`;
}
