import { useEffect, useState } from "preact/hooks";

import { api } from "../api/client";
import type { Config as ConfigData, Info } from "../api/contract";
import { Button } from "../ui/Button";
import { Message } from "../ui/Message";
import { PageHeader } from "../ui/PageHeader";
import { time } from "../ui/format";
import { Editor } from "./Editor";

export function Config({ info }: { readonly info?: Info }) {
  const [config, setConfig] = useState<ConfigData>(),
    [draft, setDraft] = useState(""),
    [editing, setEditing] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState<string>(),
    [message, setMessage] = useState(""),
    [readAt, setReadAt] = useState<number>();
  useEffect(() => {
    const controller = new AbortController();
    void api.config(controller.signal).then((result) => {
      if (controller.signal.aborted) return;
      if (result.kind === "error") setError(result.message);
      else {
        setConfig(result.value);
        setDraft(result.value.content);
        setReadAt(Date.now());
      }
    });
    return () => controller.abort();
  }, []);
  async function reload() {
    setBusy(true);
    setError(undefined);
    setMessage("");
    const result = await api.config();
    setBusy(false);
    if (result.kind === "error") setError(result.message);
    else {
      setConfig(result.value);
      setDraft(result.value.content);
      setEditing(false);
      setReadAt(Date.now());
    }
  }
  async function save() {
    if (!config || !config.writable || busy) return;
    setBusy(true);
    setError(undefined);
    setMessage("");
    const validation = await api.validate(draft);
    if (validation.kind === "error") {
      setError(validation.message);
      setBusy(false);
      return;
    }
    if (!validation.value.valid) {
      setError(
        validation.value.errors
          .map((e) => `Line ${e.line}, column ${e.column}: ${e.message}`)
          .join("\n"),
      );
      setBusy(false);
      return;
    }
    const result = await api.save(draft, config.hash);
    if (result.kind === "error") {
      setError(result.message);
      setBusy(false);
      return;
    }
    await reload();
    setMessage("Configuration saved.");
  }
  const dirty = config !== undefined && draft !== config.content;
  return (
    <main
      data-screen-label="Config"
      class="flex min-w-0 flex-1 flex-col bg-canvas"
    >
      <PageHeader
        title="Config"
        meta={
          <span class="font-mono text-body">
            {info?.config_path || "Daemon configuration"}
            {readAt ? ` · read ${time(readAt)}` : ""}
          </span>
        }
      >
        <Button
          icon="edit"
          class="h-[34px]"
          disabled={!config?.writable || busy}
          onClick={() => {
            setEditing(!editing);
            if (editing && config) setDraft(config.content);
            setError(undefined);
          }}
        >
          {editing ? "Cancel" : "Edit"}
        </Button>
        <Button
          icon="restart"
          class="h-[34px]"
          disabled={busy}
          title={
            dirty ? "Reload and discard unsaved edits" : "Reload configuration"
          }
          onClick={() => void reload()}
        >
          Reload
        </Button>
        <Button
          class="h-[34px] px-3.5 font-semibold"
          tone={dirty ? "primary" : "neutral"}
          disabled={!dirty || busy || !config?.writable}
          onClick={() => void save()}
        >
          {busy ? "Saving…" : dirty ? "Save" : "Save (no changes)"}
        </Button>
      </PageHeader>
      <div class="min-h-0 flex-1 overflow-auto px-4 pt-3.5 pb-8">
        {error && <Message error={error} />}{" "}
        {message && (
          <p role="status" class="py-2 text-[13px] text-ok">
            {message}
          </p>
        )}
        {config ? (
          <>
            {!config.writable && (
              <p class="pb-2 text-xs text-muted">
                This configuration is read only.
              </p>
            )}
            <Editor
              value={draft}
              editable={editing && config.writable && !busy}
              onChange={setDraft}
            />
          </>
        ) : (
          <Message loading={!error} empty="Configuration is unavailable." />
        )}
      </div>
    </main>
  );
}
