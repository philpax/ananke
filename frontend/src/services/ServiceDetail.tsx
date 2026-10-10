import { useState } from "preact/hooks";

import type { Range } from "../api/client";
import { api, rangeKey } from "../api/client";
import type { Device, Service } from "../api/contract";
import { useResource } from "../api/resource";
import type { CapturedEvent } from "../api/system";
import { EventRows } from "../events/EventRows";
import { eventsInRange } from "../events/data";
import { Charts } from "../history/Charts";
import { RangeControl } from "../history/RangeControl";
import { Summary } from "../history/Summary";
import { activityCharts } from "../history/data";
import type { Route } from "../layout/navigation";
import { Button } from "../ui/Button";
import { Copy } from "../ui/Copy";
import { Icon } from "../ui/Icon";
import { Message } from "../ui/Message";
import { State } from "../ui/State";
import { Facts } from "./Facts";
import { Logs } from "./Logs";
import { Placement } from "./Placement";

export function ServiceDetail({
  name,
  summary,
  devices,
  range,
  onRange,
  captured,
  onNavigate,
  onBrowse,
  singlePane,
  listClosed,
  serviceCount,
}: {
  readonly name: string;
  readonly summary?: Service;
  readonly devices: readonly Device[];
  readonly range: Range;
  readonly onRange: (r: Range) => void;
  readonly captured: readonly CapturedEvent[];
  readonly onNavigate: (r: Route) => void;
  readonly onBrowse: () => void;
  readonly singlePane: boolean;
  readonly listClosed: boolean;
  readonly serviceCount: number;
}) {
  const resource = useResource(name, api.detail(name));
  const key = `${name}:${rangeKey(range)}`;
  const metrics = useResource(key, api.metrics(range, name));
  const restarts = useResource(key, api.restarts(range, name));
  const [pending, setPending] = useState(false),
    [actionError, setActionError] = useState<string>();
  const detail = resource.data;
  const state = summary?.state ?? detail?.state ?? "idle";
  const running = state === "running",
    active = running || state === "starting" || state === "draining";
  const disabled = state.startsWith("disabled");
  const rows = eventsInRange(
    captured,
    restarts.data?.restarts ?? [],
    range,
    "all",
    name,
  );
  const buckets = metrics.data?.buckets ?? [];
  async function action(
    kind: "start" | "stop" | "restart" | "disable" | "enable",
  ) {
    if (pending) return;
    setPending(true);
    setActionError(undefined);
    const result = await api.action(name, kind);
    setPending(false);
    if (result.kind === "error") setActionError(result.message);
  }
  const openUrl = serviceUrl(
    summary?.port ?? detail?.port,
    summary?.web_ui_url,
  );
  return (
    <main
      data-screen-label="Services"
      aria-label={`Service ${name}`}
      class="min-w-0 flex-1 overflow-auto bg-canvas"
    >
      <div
        class="
          flex flex-col gap-5 px-6 pt-5 pb-9
          max-[899px]:px-3 max-[899px]:pt-3 max-[899px]:pb-[26px]
        "
      >
        {singlePane && (
          <Button icon="back" class="h-[34px] self-start" onClick={onBrowse}>
            All services
          </Button>
        )}
        {listClosed && !singlePane && (
          <button
            type="button"
            aria-expanded="false"
            class="
              inline-flex h-8 items-center gap-2 self-start text-[13px]
              font-semibold text-accent
              hover:text-accent-hover
            "
            onClick={onBrowse}
          >
            <Icon name="browse" size={15} />
            Browse services{" "}
            <span class="font-normal text-muted">{serviceCount}</span>
          </button>
        )}
        <header
          class="
            sticky top-0 z-5 flex min-h-14 flex-wrap items-center
            justify-between gap-3 border-b border-line bg-surface px-4 py-2.5
            min-[1180px]:-mx-6 min-[1180px]:-mt-5
          "
          data-service-header
        >
          <div class="flex min-w-0 flex-wrap items-center gap-[7px]">
            <div class="flex max-w-full min-w-0 items-center gap-2">
              <h1
                class="
                  min-w-0 font-mono text-base leading-[1.3] font-semibold
                  wrap-anywhere
                "
              >
                {name}
              </h1>
              <Copy text={name} />
            </div>
            <State state={state} />
          </div>
          <div class="flex flex-wrap gap-2">
            <Button
              tone="primary"
              icon="open"
              class="h-9 px-3.5 font-semibold"
              disabled={!openUrl}
              onClick={() => {
                if (openUrl)
                  window.open(openUrl, "_blank", "noopener,noreferrer");
              }}
            >
              Open
            </Button>
            <Button
              icon="chat"
              class="h-9 px-3.5"
              disabled={
                !summary?.openai_compat ||
                (summary.modality !== undefined && summary.modality !== "chat")
              }
              onClick={() => onNavigate({ section: "chat", model: name })}
            >
              Chat
            </Button>
            {running && (
              <Button
                icon="restart"
                class="h-9 px-3.5"
                disabled={pending}
                onClick={() => void action("restart")}
              >
                Restart
              </Button>
            )}
            {active ? (
              <Button
                tone="stop"
                icon="stop"
                class="h-9 px-3.5"
                disabled={pending || state === "draining"}
                onClick={() => void action("stop")}
              >
                Stop
              </Button>
            ) : (
              <Button
                tone="start"
                icon="start"
                class="h-9 px-3.5"
                disabled={pending || disabled}
                onClick={() => void action("start")}
              >
                Start
              </Button>
            )}
            <Button
              icon="disable"
              class="h-9 px-3.5"
              disabled={pending}
              onClick={() => void action(disabled ? "enable" : "disable")}
            >
              {disabled ? "Enable" : "Disable"}
            </Button>
          </div>
        </header>
        {actionError && <Message error={actionError} />}{" "}
        {resource.error && <Message error={resource.error} />}
        {detail ? (
          <div
            class="
              grid grid-cols-[repeat(auto-fit,minmax(min(100%,320px),1fr))]
              gap-4
            "
          >
            <Placement detail={detail} summary={summary} devices={devices} />
            <Facts detail={detail} summary={summary} />
          </div>
        ) : (
          <Message
            loading={resource.loading}
            empty="Service details are unavailable."
          />
        )}
        <div
          aria-label="Service history"
          class="
            flex min-w-0 flex-col gap-[18px] border-t border-line-strong pt-4
          "
        >
          <h2 class="text-[15px] font-semibold">History</h2>
          <RangeControl range={range} onChange={onRange} />
          {(metrics.error || restarts.error) && (
            <Message error={metrics.error ?? restarts.error} />
          )}
          <section
            aria-label="Summary for selected time range"
            class="min-w-0 border-t border-line pt-3"
          >
            <h2 class="mb-3 text-[13px] leading-[1.35] font-semibold">
              Window summary
            </h2>
            <Summary
              buckets={buckets}
              unavailable={metrics.loading || !!metrics.error}
            />
          </section>
          <Charts
            title="Service activity"
            charts={activityCharts(
              buckets,
              restarts.data?.restarts ?? [],
              [],
              range,
              false,
            )}
            range={range}
          />
          <section
            aria-label="Recent events for this service"
            class="min-w-0 border-t border-line pt-3"
          >
            <div class="flex flex-wrap items-center gap-3 pb-[9px]">
              <h2 class="text-[13px] leading-[1.35] font-semibold">
                Recent events
              </h2>
              <Button
                class="ml-auto h-7 bg-surface px-2.5 text-[12.5px]"
                onClick={() => onNavigate({ section: "events" })}
              >
                All events
              </Button>
            </div>
            <EventRows rows={rows.slice(0, 20)} />
            {!rows.length && (
              <Message empty="No captured events or recorded restarts in this range." />
            )}
          </section>
          <Logs name={name} range={range} running={active} />
        </div>
      </div>
    </main>
  );
}

function serviceUrl(port?: number, published?: string | null) {
  if (published) {
    try {
      const url = new URL(published);
      if (url.protocol === "http:" || url.protocol === "https:")
        return url.href;
    } catch {
      /* Invalid published URLs do not bypass the service proxy. */
    }
  }
  if (!port) return null;
  const url = new URL(location.href);
  url.protocol = "http:";
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  url.port = String(port);
  return url.href;
}
