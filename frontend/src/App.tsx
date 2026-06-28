// Root component: router + AppShell + event subscription.
// Routes are defined here; each view lives in its own component file
// under `src/components/`.

import { BrowserRouter, Routes, Route } from "react-router-dom";

import { AppShell } from "./components/layout/AppShell.tsx";
import { useEventsConnection } from "./api/events.ts";
import { DashboardView } from "./components/dashboard/DashboardView.tsx";
import { ServiceDetailView } from "./components/services/ServiceDetailView.tsx";
import { EventsView } from "./components/events/EventsView.tsx";
import { ChatView } from "./components/chat/ChatView.tsx";
import { MetricsView } from "./components/metrics/MetricsView.tsx";
import { ConfigEditorView } from "./components/config/ConfigEditorView.tsx";
import { OneshotsView } from "./components/oneshots/OneshotsView.tsx";

function Placeholder({ title }: { title: string }) {
  return (
    <div className="flex h-full items-center justify-center p-4">
      <div className="flex flex-col items-center gap-3 text-center">
        <svg
          width="32"
          height="32"
          viewBox="0 0 24 24"
          className="text-border-strong"
          aria-hidden="true"
        >
          <circle
            cx="12"
            cy="12"
            r="8"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          />
          <circle cx="12" cy="12" r="2.25" fill="currentColor" />
          <path
            d="M12 1.5v4M12 18.5v4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
        <h1 className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-secondary">
          {title}
        </h1>
        <p className="text-sm text-tertiary">Not yet instrumented.</p>
      </div>
    </div>
  );
}

export default function App() {
  // Open the events WebSocket on mount. It drives query invalidation
  // for near-instant updates.
  useEventsConnection();

  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<DashboardView />} />
          <Route path="/services/:name" element={<ServiceDetailView />} />
          <Route path="/chat" element={<ChatView />} />
          <Route path="/oneshots" element={<OneshotsView />} />
          <Route path="/config" element={<ConfigEditorView />} />
          <Route path="/events" element={<EventsView />} />
          <Route path="/stats" element={<MetricsView />} />
          <Route path="*" element={<Placeholder title="Not found" />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}
