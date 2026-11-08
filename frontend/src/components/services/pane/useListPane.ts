// The service list pane's stored width and whether it is hidden on wide
// screens.

import { useStoredValue } from "../../../hooks/useStoredValue.ts";
import { clampListWidth, parseListPrefs } from "./listPane.ts";

export type ListPane = {
  readonly width: number;
  readonly hidden: boolean;
  readonly setWidth: (width: number) => void;
  readonly setHidden: (hidden: boolean) => void;
};

export function useListPane(): ListPane {
  const [width, writeWidth] = useStoredValue(
    WIDTH_KEY,
    (raw) => parseListPrefs(raw, null).width,
  );
  const [hidden, writeHidden] = useStoredValue(
    HIDDEN_KEY,
    (raw) => parseListPrefs(null, raw).hidden,
  );
  return {
    width,
    hidden,
    setWidth: (next) => writeWidth(String(clampListWidth(next))),
    setHidden: (next) => writeHidden(String(next)),
  };
}

const WIDTH_KEY = "ananke-services-list-width";
const HIDDEN_KEY = "ananke-services-list-hidden";
