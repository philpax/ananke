import { useEffect, useState } from "preact/hooks";

import { useSystem } from "./api/system";
import { Chat } from "./chat/Chat";
import { useChat } from "./chat/useChat";
import { Config } from "./config/Config";
import { Events } from "./events/Events";
import { Stats } from "./history/Stats";
import { presetRange } from "./history/range";
import { Hardware } from "./layout/Hardware";
import { Nav } from "./layout/Nav";
import { useNavigation } from "./layout/navigation";
import { useViewport } from "./layout/viewport";
import { Services } from "./services/Services";

export function App() {
  const system = useSystem(),
    width = useViewport(),
    { route, navigate } = useNavigation(),
    chat = useChat();
  const [lastService, setLastService] = useState(route.service);
  const [configVisited, setConfigVisited] = useState(
    route.section === "config",
  );
  useEffect(() => {
    if (route.section === "services") setLastService(route.service);
    if (route.section === "config") setConfigVisited(true);
  }, [route]);
  const [range, setRange] = useState(() => presetRange(6));
  useEffect(() => {
    const timer = setInterval(
      () =>
        setRange((previous) =>
          previous.preset ? presetRange(previous.preset) : previous,
        ),
      15_000,
    );
    return () => clearInterval(timer);
  }, []);
  const services = system.services.data?.services ?? [],
    devices = system.devices.data ?? [];
  const chatModel = services.find(
    (service) =>
      service.name === lastService &&
      service.openai_compat &&
      (!service.modality || service.modality === "chat"),
  )?.name;
  const compact = width < 900;
  const nav = (
    <Nav
      current={route.section}
      compact={compact}
      onNavigate={(section) =>
        navigate({
          section,
          service: section === "services" ? lastService : undefined,
          model: section === "chat" ? chatModel : undefined,
        })
      }
    />
  );
  return (
    <div
      class="
        flex h-dvh w-full flex-col overflow-hidden bg-canvas font-sans text-sm
        leading-[normal] text-ink
      "
    >
      <Hardware
        devices={devices}
        info={system.info.data}
        compact={compact}
        error={system.devices.error}
      />
      {compact && nav}
      <div class="relative flex min-h-0 min-w-0 flex-1">
        {!compact && nav}
        <div
          class={
            route.section === "services"
              ? "flex min-h-0 min-w-0 flex-1"
              : "hidden"
          }
        >
          <Services
            active={route.section === "services"}
            services={services}
            devices={devices}
            selected={
              route.section === "services" ? route.service : lastService
            }
            width={width}
            range={range}
            onRange={setRange}
            onNavigate={navigate}
            captured={system.events}
            loading={system.services.loading}
            error={system.services.error}
          />
        </div>
        {route.section === "chat" && (
          <Chat
            chat={chat}
            services={services}
            requestedModel={route.model}
            active
          />
        )}
        {route.section === "events" && (
          <Events
            range={range}
            onRange={setRange}
            captured={system.events}
            capturedSince={system.capturedSince}
            connected={system.connected}
            onNavigate={navigate}
          />
        )}
        {route.section === "stats" && (
          <Stats range={range} onRange={setRange} />
        )}{" "}
        {(configVisited || route.section === "config") && (
          <div
            class={
              route.section === "config"
                ? "flex min-h-0 min-w-0 flex-1"
                : "hidden"
            }
          >
            <Config info={system.info.data} />
          </div>
        )}
      </div>
    </div>
  );
}
