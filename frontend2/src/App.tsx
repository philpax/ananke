// Root component: router + AppShell + event subscription.
// Routes are defined here; each view lives in its own component file
// under `src/components/`.

import { BrowserRouter, Routes, Route } from "react-router-dom";

import { AppShell } from "./components/layout/AppShell.tsx";
import { useEvents } from "./api/events.ts";

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
          <Route path="/" element={<Placeholder title="Dashboard" />} />
          <Route path="/services" element={<Placeholder title="Services" />} />
          <Route
            path="/services/:name"
            element={<Placeholder title="Service detail" />}
          />
          <Route path="/devices" element={<Placeholder title="Devices" />} />
          <Route path="/chat" element={<Placeholder title="Chat" />} />
          <Route path="/oneshots" element={<Placeholder title="Oneshots" />} />
          <Route path="/config" element={<Placeholder title="Config" />} />
          <Route path="/events" element={<Placeholder title="Events" />} />
          <Route path="/metrics" element={<Placeholder title="Metrics" />} />
          <Route path="*" element={<Placeholder title="Not found" />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}
