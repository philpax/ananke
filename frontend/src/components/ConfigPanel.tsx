import { useConfig } from "../api/hooks.ts";

export function ConfigPanel() {
  const { data, error, isPending } = useConfig();

  if (isPending)
    return <section className="opacity-60">Loading config…</section>;
  if (error)
    return <section className="text-red-600">Config: {error.message}</section>;

  return (
    <section>
      <h2 className="text-lg font-semibold mb-2">Config</h2>
      <details className="border border-gray-300 rounded">
        <summary className="cursor-pointer p-2 text-sm text-gray-700">
          Raw TOML (hash {data.hash.slice(0, 12)}…)
        </summary>
        <pre className="bg-gray-50 text-xs p-2 overflow-x-auto max-h-96 whitespace-pre-wrap">
          {data.content}
        </pre>
      </details>
    </section>
  );
}
