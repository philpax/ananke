import { useEffect, useRef, useState } from "preact/hooks";

import type { Range } from "../api/client";
import { api } from "../api/client";
import { connectionStatus, watch } from "../api/socket";
import type { LogLine } from "../api/contract";
import { Copy } from "../ui/Copy";
import { Message } from "../ui/Message";
import { time } from "../ui/format";

export function Logs({
  name,
  range,
  running,
}: {
  readonly name: string;
  readonly range: Range;
  readonly running: boolean;
}) {
  const [lines, setLines] = useState<readonly LogLine[]>([]),
    [cursor, setCursor] = useState<string | null>(null),
    [error, setError] = useState<string>(),
    [loading, setLoading] = useState(true),
    [streaming, setStreaming] = useState(false);
  const ref = useRef<HTMLDivElement>(null),
    busyRef = useRef(false),
    cursorRef = useRef<string | null>(null),
    generationRef = useRef(0),
    retainedRef = useRef<readonly LogLine[]>([]);
  const live = range.preset !== null;
  const since = live ? 0 : range.since,
    until = live ? 0 : range.until,
    preset = range.preset;
  useEffect(() => {
    cursorRef.current = cursor;
  }, [cursor]);
  useEffect(() => {
    const controller = new AbortController();
    const run = ++generationRef.current;
    let initial = true;
    let backfilling = false;
    setLines([]);
    setCursor(null);
    cursorRef.current = null;
    setLoading(true);
    setError(undefined);
    busyRef.current = false;
    const currentRange = () => {
      const now = Date.now();
      return preset
        ? { preset, since: now - preset * 3_600_000, until: now }
        : { preset: null, since, until };
    };
    retainedRef.current = [];
    const stop = watch(api.logSource(name, currentRange()), (result) => {
      if (controller.signal.aborted || run !== generationRef.current) return;
      setLoading(false);
      if (result.kind === "error") {
        setError(result.message);
        return;
      }
      setError(undefined);
      const el = ref.current;
      const atBottom =
        initial ||
        (el ? el.scrollHeight - el.clientHeight - el.scrollTop < 40 : false);
      const previous = retainedRef.current;
      retainedRef.current = mergeLogs(
        retainedRef.current,
        result.value.logs,
      ).slice(-5000);
      setLines(retainedRef.current);
      if (initial) setCursor(result.value.next_cursor ?? null);
      initial = false;
      if (atBottom)
        requestAnimationFrame(() => {
          if (el) el.scrollTop = el.scrollHeight;
        });
      // A reconnect or a large committed batch can exceed the first history page.
      // Follow opaque cursors until the new page overlaps our retained tail.
      if (
        !backfilling &&
        previous.length &&
        result.value.next_cursor &&
        result.value.logs.length &&
        !result.value.logs.some((line) =>
          previous.some(
            (old) => old.run_id === line.run_id && old.seq === line.seq,
          ),
        )
      ) {
        backfilling = true;
        void (async () => {
          let before = result.value.next_cursor;
          let count = 0;
          while (before && count < 5000 && !controller.signal.aborted) {
            const page = await api.logs(
              name,
              currentRange(),
              before,
              controller.signal,
            );
            if (controller.signal.aborted || run !== generationRef.current)
              return;
            if (page.kind === "error") {
              setError(page.message);
              break;
            }
            retainedRef.current = mergeLogs(
              retainedRef.current,
              page.value.logs,
            ).slice(-5000);
            setLines(retainedRef.current);
            count += page.value.logs.length;
            if (
              page.value.logs.some((line) =>
                previous.some(
                  (old) => old.run_id === line.run_id && old.seq === line.seq,
                ),
              )
            )
              break;
            if (page.value.next_cursor === before || !page.value.logs.length) {
              setError(
                "Logs: the server returns a history cursor that does not advance.",
              );
              break;
            }
            before = page.value.next_cursor;
          }
          backfilling = false;
        })();
      }
    });
    return () => {
      controller.abort();
      stop();
    };
  }, [name, since, until, preset]);
  useEffect(
    () =>
      connectionStatus((connected) =>
        setStreaming(connected && live && running),
      ),
    [live, running],
  );

  async function older() {
    if (busyRef.current || !cursorRef.current) return;
    busyRef.current = true;
    const run = generationRef.current,
      el = ref.current,
      previousHeight = el?.scrollHeight ?? 0;
    const result = await api.logs(
      name,
      live ? { ...range, until: Date.now() } : range,
      cursorRef.current,
    );
    if (run !== generationRef.current) return;
    busyRef.current = false;
    if (result.kind === "error") {
      setError(result.message);
      return;
    }
    retainedRef.current = mergeLogs(retainedRef.current, result.value.logs);
    setLines(retainedRef.current);
    setCursor(result.value.next_cursor ?? null);
    requestAnimationFrame(() => {
      if (el) el.scrollTop = el.scrollHeight - previousHeight;
    });
  }
  const text = lines
    .map((l) => `${time(l.timestamp_ms)}  ${l.line}`)
    .join("\n");
  return (
    <section class="min-w-0 border-t border-line pt-3">
      <div class="flex flex-wrap items-center gap-2.5 pb-[9px]">
        <h3 class="text-[13px] leading-[1.35] font-semibold">Service log</h3>
        <span class="text-xs text-muted">
          {lines.length} lines ·{" "}
          {streaming ? "streaming" : live ? "captured" : "selected range"}
        </span>
        <span class="ml-auto flex">
          <Copy text={text} label="service log" />
        </span>
      </div>
      {error && <Message error={error} />}
      <div
        ref={ref}
        role="log"
        aria-label="Service log viewer"
        tabIndex={0}
        onScroll={(e) => {
          if (e.currentTarget.scrollTop < 20) void older();
        }}
        class="
          max-h-[230px] min-h-[180px] resize-y overflow-auto border border-line
          bg-[#0a0b0d] p-3 font-mono text-[12.5px] leading-[1.75] text-body
        "
      >
        {!lines.length && (
          <Message loading={loading} empty="No log lines in this range." />
        )}
        {lines.map((l) => (
          <div
            key={`${l.run_id}:${l.seq}`}
            class={
              l.stream === "stderr"
                ? `wrap-anywhere whitespace-pre-wrap text-soft`
                : `wrap-anywhere whitespace-pre-wrap`
            }
          >
            {time(l.timestamp_ms)} {l.line}
          </div>
        ))}
      </div>
      {cursor && (
        <span class="block py-2 text-xs text-muted">
          Scroll up for older lines
        </span>
      )}
    </section>
  );
}

function mergeLogs(a: readonly LogLine[], b: readonly LogLine[]) {
  const rows = new Map(a.map((l) => [`${l.run_id}:${l.seq}`, l]));
  for (const l of b) rows.set(`${l.run_id}:${l.seq}`, l);
  return [...rows.values()].sort(
    (a, b) =>
      a.timestamp_ms - b.timestamp_ms || a.run_id - b.run_id || a.seq - b.seq,
  );
}
