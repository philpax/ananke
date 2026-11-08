// Root component: router + AppShell + event-driven invalidation.
// Routes are defined here; each view lives in its own component file
// under `src/components/`.

import { BrowserRouter, Routes, Route } from "react-router-dom";

import { AppShell } from "./components/layout/AppShell.tsx";
import { useEventInvalidation } from "./api/live/invalidation.ts";
import { ServicesView } from "./components/services/ServicesView.tsx";
import { EventsView } from "./components/events/EventsView.tsx";
import { ChatView } from "./components/chat/ChatView.tsx";
import { MetricsView } from "./components/metrics/MetricsView.tsx";
import { ConfigEditorView } from "./components/config/ConfigEditorView.tsx";
import { NotFound } from "./components/layout/NotFound.tsx";

export default function App() {
  // Live queries update themselves; this refreshes the few that are not
  // live (config, restarts) when an event says they changed.
  useEventInvalidation();

  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          {/* One element type at one position for both, so React keeps
              the list mounted while services are opened. */}
          <Route path="/" element={<ServicesView />} />
          <Route path="/services/:name" element={<ServicesView />} />
          <Route path="/chat" element={<ChatView />} />
          <Route path="/config" element={<ConfigEditorView />} />
          <Route path="/events" element={<EventsView />} />
          <Route path="/stats" element={<MetricsView />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}
