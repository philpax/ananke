// The logs viewer's controls: stream, search, and run filters, the window's
// status, and the follow toggle. Wraps onto several rows on a phone.

import { useTranslation } from "react-i18next";

import { Button } from "../ui/button/Button.tsx";
import { Select } from "../ui/form/Select.tsx";
import { SegmentedToggle } from "../ui/form/SegmentedToggle.tsx";
import { TextInput } from "../ui/form/TextInput.tsx";
import type { StreamFilter } from "./logFilter.ts";

export type LogStatus = {
  /// A fixed range shows a selected window and does not follow.
  readonly fixed: boolean;
  readonly live: boolean;
  readonly skipped: number;
  readonly truncated: boolean;
  readonly lineCount: number;
};

export function LogToolbar({
  status,
  stream,
  onStream,
  search,
  onSearch,
  runIds,
  run,
  onRun,
  pinned,
  onFollow,
}: {
  readonly status: LogStatus;
  readonly stream: StreamFilter;
  readonly onStream: (stream: StreamFilter) => void;
  readonly search: string;
  readonly onSearch: (search: string) => void;
  readonly runIds: readonly number[];
  readonly run: number | null;
  readonly onRun: (run: number | null) => void;
  readonly pinned: boolean;
  readonly onFollow: () => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-2">
      <SegmentedToggle<StreamFilter>
        label={t("logs.streamFilter")}
        options={STREAMS.map((s) => ({ label: t(`logs.${s}`), value: s }))}
        selected={stream}
        onChange={onStream}
      />
      <TextInput
        type="search"
        size="sm"
        aria-label={t("logs.searchLabel")}
        placeholder={t("logs.search")}
        value={search}
        onChange={(e) => onSearch(e.target.value)}
        className="w-full text-xs sm:w-44"
      />
      {runIds.length > 1 && (
        <Select
          size="sm"
          aria-label={t("logs.runFilter")}
          className="text-xs"
          value={run ?? ""}
          onChange={(e) =>
            onRun(e.target.value ? Number(e.target.value) : null)
          }
        >
          <option value="">{t("logs.allRuns")}</option>
          {runIds.map((r) => (
            <option key={r} value={r}>
              {t("logs.run", { value: r })}
            </option>
          ))}
        </Select>
      )}
      <div
        role="status"
        className="ml-auto flex flex-wrap items-center gap-2 text-xs text-muted"
      >
        {status.fixed ? (
          <span>{t("logs.selectedRange")}</span>
        ) : (
          status.live && <span className="text-ok">{t("logs.live")}</span>
        )}
        {status.skipped > 0 && (
          <span className="text-warn">
            {t("logs.dropped", { value: status.skipped })}
          </span>
        )}
        {status.truncated && (
          <span className="text-warn">{t("logs.truncated")}</span>
        )}
        <span className="font-mono">
          {t("logs.lines", { value: status.lineCount })}
        </span>
      </div>
      <Button
        tone="quiet"
        size="sm"
        aria-pressed={pinned}
        className={pinned ? "text-accent-text" : undefined}
        onClick={onFollow}
      >
        {t(status.fixed ? "logs.jumpToEnd" : "logs.follow")}
      </Button>
    </div>
  );
}

const STREAMS: readonly StreamFilter[] = [
  "both",
  "stdout",
  "stderr",
  "combined",
];
