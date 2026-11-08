// Y-axis tick labels in compact notation ("25k", "2.2M"), so large values
// fit the fixed axis width instead of being clipped to their tail.

export function compactTick(value: number, locale?: string): string {
  return compactFormat(locale).format(value);
}

const formats = new Map<string, Intl.NumberFormat>();

function compactFormat(locale: string | undefined): Intl.NumberFormat {
  const key = locale ?? "";
  let format = formats.get(key);
  if (!format) {
    format = new Intl.NumberFormat(locale, {
      notation: "compact",
      maximumFractionDigits: 1,
    });
    formats.set(key, format);
  }
  return format;
}
