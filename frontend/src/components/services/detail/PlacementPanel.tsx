// Where the service's memory goes: the constraints its config sets, the
// placement it would take now and whether that fits, and the pledge it
// holds or the override it declares.

import { useTranslation } from "react-i18next";

import type {
  DevicePlacement,
  DeviceShortfall,
  PlacementPreview,
  ServiceDetail,
} from "../../../api/client.ts";
import { formatBytes } from "../../../util.ts";
import { FactList } from "../../ui/data/FactList.tsx";
import { MemoryBar } from "../../ui/data/MemoryBar.tsx";
import { Panel } from "../../ui/layout/Panel.tsx";
import { Chip } from "../../ui/status/Chip.tsx";
import type { Tone } from "../../ui/tone.ts";
import { constraintRows } from "./placementConstraints.ts";

export function PlacementPanel({ detail }: { readonly detail: ServiceDetail }) {
  const { t } = useTranslation();
  const preview = detail.placement_preview;
  const pledge = Object.entries(detail.current_allocation);
  const override = Object.entries(detail.placement_override);
  return (
    <Panel heading={t("serviceDetail.placement")}>
      <div className="flex flex-col gap-4">
        <FactList facts={constraintRows(detail.placement, t)} />
        {preview && <Preview preview={preview} />}
        {(pledge.length > 0 || override.length > 0) && (
          <FactList
            facts={[
              pledge.length > 0 && {
                label: t("serviceDetail.currentPledge"),
                value: slotList(pledge),
                mono: true,
              },
              override.length > 0 && {
                label: t("serviceDetail.configuredOverride"),
                value: slotList(override),
                title: t("serviceDetail.configuredOverrideTitle"),
                mono: true,
              },
            ]}
          />
        )}
      </div>
    </Panel>
  );
}

function Preview({ preview }: { readonly preview: PlacementPreview }) {
  const { t } = useTranslation();
  const { verdict } = preview;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <h3 className="text-xs text-muted">{t("serviceDetail.preview")}</h3>
        <Chip tone={VERDICT_TONE[verdict.kind]}>
          {t(VERDICT_LABEL[verdict.kind])}
        </Chip>
      </div>
      {preview.devices.length > 0 ? (
        <ul className="flex flex-col gap-1.5">
          {preview.devices.map((device) => (
            <li key={device.device}>
              <PlacementBar
                device={device}
                needsEviction={verdict.kind === "needs_eviction"}
              />
            </li>
          ))}
        </ul>
      ) : (
        <div className="flex flex-col gap-1 text-xs text-muted">
          <p>{t("serviceDetail.noPlacementFits")}</p>
          {verdict.kind === "does_not_fit" && (
            <Shortfalls shortfalls={verdict.shortfalls} />
          )}
        </div>
      )}
      {preview.expert_offload_bytes > 0 && (
        <p className="text-xs text-muted">
          {t("serviceDetail.expertOffload", {
            layers: preview.expert_offload_layers,
            bytes: formatBytes(preview.expert_offload_bytes),
            count: preview.expert_offload_layers,
          })}
        </p>
      )}
    </div>
  );
}

// Naming the device matters: the binding constraint is often host RAM
// rather than a GPU, and the ids match `GET /api/devices`.
function Shortfalls({
  shortfalls,
}: {
  readonly shortfalls: readonly DeviceShortfall[];
}) {
  const { t } = useTranslation();
  if (shortfalls.length === 0) {
    return <p>{t("serviceDetail.noEligibleDevices")}</p>;
  }
  return (
    <ul className="font-mono">
      {shortfalls.map((s) => (
        <li key={s.device}>
          {t("serviceDetail.shortfall", {
            device: s.device,
            requested: formatBytes(s.requested_bytes),
            available: formatBytes(s.available_bytes),
          })}
        </li>
      ))}
    </ul>
  );
}

function PlacementBar({
  device,
  needsEviction,
}: {
  readonly device: DevicePlacement;
  readonly needsEviction: boolean;
}) {
  const { t } = useTranslation();
  const { bytes, max_bytes, used_by_others_bytes, total_bytes } = device;
  const canGrow = max_bytes > bytes;
  return (
    <div className="text-xs">
      <div className="flex items-baseline justify-between gap-2 font-mono">
        <span className="text-ink">{device.device}</span>
        <span className="text-muted">
          {formatBytes(bytes)}
          {canGrow && <>&ndash;{formatBytes(max_bytes)}</>}
          {total_bytes > 0 && <> / {formatBytes(total_bytes)}</>}
        </span>
      </div>
      {total_bytes > 0 && (
        <MemoryBar
          variant="placement"
          total={total_bytes}
          others={used_by_others_bytes}
          self={bytes}
          growth={canGrow ? max_bytes - bytes : 0}
          needsEviction={needsEviction}
          label={t("serviceDetail.placementBar", {
            self: formatBytes(bytes),
            others: formatBytes(used_by_others_bytes),
            total: formatBytes(total_bytes),
          })}
          className="mt-0.5"
        />
      )}
    </div>
  );
}

/// Slot → MiB pairs, as the pledge and the override both report them.
function slotList(entries: readonly [string, number][]): string {
  return entries
    .map(([slot, mib]) => `${slot} ${formatBytes(mib * 1024 * 1024)}`)
    .join(" · ");
}

const VERDICT_TONE: Readonly<
  Record<PlacementPreview["verdict"]["kind"], Tone>
> = { fits: "ok", needs_eviction: "warn", does_not_fit: "danger" };

const VERDICT_LABEL: Readonly<
  Record<PlacementPreview["verdict"]["kind"], string>
> = {
  fits: "serviceDetail.fitsNow",
  needs_eviction: "serviceDetail.needsEviction",
  does_not_fit: "serviceDetail.doesNotFit",
};
