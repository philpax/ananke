import { useState } from "react";

import { ConfigPanel } from "./components/ConfigPanel.tsx";
import { DevicesPanel } from "./components/DevicesPanel.tsx";
import { ServiceDetailPanel } from "./components/ServiceDetail.tsx";
import { ServicesTable } from "./components/ServicesTable.tsx";

function App() {
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-white text-gray-900 p-4 md:p-6 max-w-6xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">ananke</h1>
        <p className="text-sm text-gray-500">
          {window.location.host} · dashboard
        </p>
      </header>

      <div className="space-y-6">
        <DevicesPanel />
        <ServicesTable selected={selected} onSelect={setSelected} />
        <ServiceDetailPanel name={selected} />
        <ConfigPanel />
      </div>
    </div>
  );
}

export default App;
