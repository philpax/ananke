// The app's sections and how URLs map onto them. Services owns two path
// shapes (the list at `/` and one service at `/services/:name`), so
// activeness is decided here rather than by `NavLink`'s own matching.

import { isChatModel } from "../../../api/chatModels.ts";
import type { ServiceSummary } from "../../../api/client.ts";

export type Section =
  | "services"
  | "reservations"
  | "chat"
  | "events"
  | "stats"
  | "config";

export const SECTIONS: readonly Section[] = [
  "services",
  "reservations",
  "chat",
  "events",
  "stats",
  "config",
];

/// The section a path belongs to, or `undefined` for an unknown path.
export function sectionForPath(pathname: string): Section | undefined {
  const path = trimTrailingSlash(pathname);
  if (path === "" || path.startsWith(SERVICE_PREFIX)) return "services";
  const name = path.slice(1);
  return SECTIONS.find((section) => section !== "services" && section === name);
}

/// The service a `/services/:name` path names. A malformed escape falls
/// back to `undefined`, so the caller shows the list rather than throwing.
export function serviceFromPath(pathname: string): string | undefined {
  const path = trimTrailingSlash(pathname);
  if (!path.startsWith(SERVICE_PREFIX)) return undefined;
  const raw = path.slice(SERVICE_PREFIX.length);
  if (raw === "" || raw.includes("/")) return undefined;
  try {
    return decodeURIComponent(raw);
  } catch {
    return undefined;
  }
}

export type LastService = Pick<
  ServiceSummary,
  "name" | "modality" | "openai_compat"
>;

/// Where a section's nav item points. Services reopens the last service,
/// except from within Services, where it leads back to the list (on a
/// phone, the only other way back is the page's own link). Chat preselects
/// the last service when it is a chat model. `range` is the current time
/// range's query (`rangeSearch`), carried into the sections that read
/// it, Services, Events, and Stats, so moving between them keeps the range.
export function sectionHref(
  section: Section,
  lastService: LastService | undefined,
  range = "",
  current?: Section,
): string {
  switch (section) {
    case "services":
      return lastService === undefined || current === "services"
        ? `/${range}`
        : `${SERVICE_PREFIX}${encodeURIComponent(lastService.name)}${range}`;
    case "chat":
      return lastService !== undefined && isChatModel(lastService)
        ? `/chat?model=${encodeURIComponent(lastService.name)}`
        : "/chat";
    case "events":
    case "stats":
      return `/${section}${range}`;
    case "reservations":
      return "/reservations";
    case "config":
      return "/config";
    default: {
      const unreachable: never = section;
      return unreachable;
    }
  }
}

const SERVICE_PREFIX = "/services/";

function trimTrailingSlash(pathname: string): string {
  return pathname.replace(/\/+$/, "");
}
