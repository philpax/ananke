// Root component: router + AppShell + event subscription.
// Routes are defined here; each view lives in its own component file
// under `src/components/`.

import { BrowserRouter, Routes, Route } from "react-router-dom";

import { AppShell } from "./components/layout/AppShell.tsx";
import { useEvents } from "./api/events.ts";
import { DashboardView } from "./components/dashboard/DashboardView.tsx";
import { ServiceDetailView } from "./components/services/ServiceDetailView.tsx";
import { EventsView } from "./components/events/EventsView.tsx";
import { ChatView } from "./components/chat/ChatView.tsx";

function Placeholder({ title }: { title: string }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="text-center">
        <h1 className="text-lg font-medium text-primary">{title}</h1>
        <p className="mt-1 text-sm text-tertiary">coming soon</p>
      </div>
    </div>
  );
}

export default function App() {
  // Open the events WebSocket on mount. It drives query invalidation
  // for near-instant updates.
  useEvents();

  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<DashboardView />} />
          <Route path="/services/:name" element={<ServiceDetailView />} />
          <Route path="/devices" element={<Placeholder title="Devices" />} />
          <Route path="/chat" element={<ChatView />} />
          <Route path="/oneshots" element={<Placeholder title="Oneshots" />} />
          <Route path="/config" element={<Placeholder title="Config" />} />
          <Route path="/events" element={<EventsView />} />
          <Route path="/metrics" element={<Placeholder title="Metrics" />} />
          <Route path="*" element={<Placeholder title="Not found" />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}
