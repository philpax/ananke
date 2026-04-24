import { ConfigPanel } from "./components/ConfigPanel.tsx";
import { DevicesPanel } from "./components/DevicesPanel.tsx";
import { ServicesTable } from "./components/ServicesTable.tsx";

function App() {
  return (
    <div className="min-h-screen bg-white dark:bg-gray-950">
      <div className="p-4 md:p-6 max-w-6xl mx-auto text-gray-900 dark:text-gray-200">
        <header className="mb-6">
          <h1 className="text-2xl font-bold dark:text-white">ananke</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {window.location.host} · dashboard
          </p>
        </header>

        <div className="space-y-6">
          <DevicesPanel />
          <ServicesTable />
          <ConfigPanel />
        </div>
      </div>
    </div>
  );
}

export default App;
