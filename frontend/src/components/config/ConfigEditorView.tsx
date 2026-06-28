// Config editor (`/config`). A CodeMirror 6 TOML editor with debounced
// validation, ETag-gated save, reload, and a read-only toggle. Validation
// errors from the backend are shown in a panel below the editor with
// line/column references. A hash mismatch (412) surfaces a reload dialog
// so the operator can pull the server's current version.

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { EditorState, Compartment } from "@codemirror/state";
import {
  EditorView,
  lineNumbers,
  highlightActiveLine,
  highlightActiveLineGutter,
  keymap,
} from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import {
  StreamLanguage,
  syntaxHighlighting,
  HighlightStyle,
} from "@codemirror/language";
import { tags } from "@lezer/highlight";
import { toml } from "@codemirror/legacy-modes/mode/toml";

import {
  useConfig,
  useValidateConfig,
  useSaveConfig,
} from "../../api/hooks.ts";
import type { ValidationError } from "../../api/client.ts";
import { Card } from "../ui/Card.tsx";
import { ViewHeader } from "../ui/ViewHeader.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { Badge } from "../ui/Badge.tsx";

const VALIDATE_DEBOUNCE_MS = 600;

// Custom TOML highlight style using the console's theme tokens so the
// editor reads as part of the instrument panel, not a generic code
// editor. CSS variables are resolved by the browser, so colours adapt
// to dark/light mode automatically.
const tomlHighlightStyle = HighlightStyle.define([
  {
    tag: tags.comment,
    color: "var(--color-text-tertiary)",
    fontStyle: "italic",
  },
  { tag: tags.string, color: "var(--color-success)" },
  { tag: tags.number, color: "var(--color-warning)" },
  { tag: tags.bool, color: "var(--color-vision)" },
  { tag: tags.keyword, color: "var(--color-vision)" },
  { tag: tags.propertyName, color: "var(--color-accent)" },
  { tag: tags.definition(tags.propertyName), color: "var(--color-accent)" },
  { tag: tags.variableName, color: "var(--color-text-primary)" },
  { tag: tags.atom, color: "var(--color-vision)" },
  { tag: tags.punctuation, color: "var(--color-text-secondary)" },
  { tag: tags.bracket, color: "var(--color-text-secondary)" },
]);

export function ConfigEditorView() {
  const { t } = useTranslation();
  const config = useConfig();
  const validateMut = useValidateConfig();
  const saveMut = useSaveConfig();
  const [content, setContent] = useState("");
  const [originalHash, setOriginalHash] = useState("");
  const [readOnly, setReadOnly] = useState(false);
  const [hashMismatch, setHashMismatch] = useState<string | null>(null);
  const [loadedHash, setLoadedHash] = useState("");
  const [saveError, setSaveError] = useState<string | null>(null);

  const dirty =
    content !== "" && config.data && content !== config.data.content
      ? true
      : false;

  const serverReadOnly = config.data ? !config.data.writable : false;

  // Sync local state when server data changes (initial load, reload, save).
  // Using the "adjust state during render" pattern endorsed by React docs
  // instead of a useEffect, which would trigger cascading renders.
  const serverHash = config.data?.hash ?? "";
  if (serverHash !== "" && serverHash !== loadedHash) {
    setLoadedHash(serverHash);
    setContent(config.data?.content ?? "");
    setOriginalHash(serverHash);
    setHashMismatch(null);
    setReadOnly(serverReadOnly);
  }

  // Debounced validation: fire when content changes, after a short idle.
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!content || !dirty) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      validateMut.mutate(content);
    }, VALIDATE_DEBOUNCE_MS);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [content, dirty, validateMut]);

  // Clear save error when the user starts editing again.
  const handleChange = useCallback((value: string) => {
    setContent(value);
    setSaveError(null);
  }, []);

  const handleSave = useCallback(() => {
    if (!dirty || !originalHash) return;
    setSaveError(null);
    saveMut.mutate(
      { content, hash: originalHash },
      {
        onSuccess: (result) => {
          if (result.kind === "hash_mismatch") {
            setHashMismatch(result.serverHash);
          } else if (result.kind === "error") {
            setSaveError(result.message);
            // If the persist failed, switch to read-only so the
            // user doesn't keep trying to save.
            setReadOnly(true);
          }
        },
        onError: (err) => {
          setSaveError(err.message);
        },
      },
    );
  }, [content, dirty, originalHash, saveMut]);

  const handleReload = useCallback(() => {
    setHashMismatch(null);
    setSaveError(null);
    void config.refetch();
  }, [config]);

  const validationErrors =
    validateMut.data?.valid === false ? validateMut.data.errors : [];
  const saveErrors =
    saveMut.data?.kind === "validation_errors" ? saveMut.data.errors : [];
  const allErrors = [...validationErrors, ...saveErrors];

  if (config.isPending && !config.data) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  if (config.error) {
    return (
      <div className="p-4 text-sm text-danger">{config.error.message}</div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <ViewHeader>
        <h1 className="eyebrow !text-primary">{t("nav.config")}</h1>
        {readOnly && <Badge variant="neutral">read-only</Badge>}
        {dirty && <Badge variant="warning">unsaved</Badge>}
        {saveMut.isPending && <Badge variant="accent">saving…</Badge>}
        <div className="ml-auto flex items-center gap-2">
          {!serverReadOnly && (
            <button
              type="button"
              onClick={() => setReadOnly((r) => !r)}
              className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                readOnly
                  ? "bg-elevated text-primary"
                  : "text-tertiary hover:text-secondary"
              }`}
            >
              {readOnly ? "Read-only" : "Editable"}
            </button>
          )}
          <button
            type="button"
            onClick={handleReload}
            disabled={config.isFetching}
            className="rounded-md bg-elevated px-2.5 py-1 text-xs text-primary transition-colors hover:bg-border-strong disabled:opacity-40"
          >
            Reload
          </button>
          {!serverReadOnly && (
            <button
              type="button"
              onClick={handleSave}
              disabled={!dirty || saveMut.isPending || readOnly}
              className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-[var(--color-base)] transition-colors hover:bg-accent/90 disabled:opacity-40"
            >
              Save
            </button>
          )}
        </div>
      </ViewHeader>

      {saveError && (
        <div className="shrink-0 border-b border-border-default bg-danger/10 px-4 py-2 text-xs text-danger">
          {saveError}
        </div>
      )}

      <div className="flex-1 overflow-hidden">
        <CodeMirrorEditor
          content={content}
          readOnly={readOnly}
          onChange={handleChange}
        />
      </div>

      {allErrors.length > 0 && <ValidationPanel errors={allErrors} />}

      {hashMismatch && (
        <HashMismatchDialog
          serverHash={hashMismatch}
          onReload={handleReload}
          onDismiss={() => setHashMismatch(null)}
        />
      )}
    </div>
  );
}

function CodeMirrorEditor({
  content,
  readOnly,
  onChange,
}: {
  content: string;
  readOnly: boolean;
  onChange: (value: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const readOnlyComp = useRef(new Compartment());
  const onChangeRef = useRef(onChange);
  const isExternalUpdate = useRef(false);
  const initialContent = useRef(content);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  // Create the editor once on mount.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const theme = EditorView.theme({
      "&": {
        backgroundColor: "var(--color-surface)",
        color: "var(--color-text-primary)",
        height: "100%",
        fontSize: "13px",
      },
      ".cm-scroller": {
        fontFamily: "'IBM Plex Mono', monospace",
        overflow: "auto",
      },
      ".cm-content": { padding: "8px 0" },
      ".cm-gutters": {
        backgroundColor: "var(--color-surface)",
        color: "var(--color-text-tertiary)",
        border: "none",
        borderRight: "1px solid var(--color-border-default)",
      },
      ".cm-lineNumbers .cm-gutterElement": {
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: "11px",
        padding: "0 8px 0 12px",
      },
      ".cm-activeLine": { backgroundColor: "var(--color-elevated)" },
      ".cm-activeLineGutter": {
        backgroundColor: "var(--color-elevated)",
        color: "var(--color-text-secondary)",
      },
      "&.cm-focused .cm-selectionBackground, ::selection": {
        backgroundColor: "rgba(139,124,248,0.2)",
      },
      ".cm-cursor, .cm-dropCursor": {
        borderLeftColor: "var(--color-accent)",
      },
    });

    const state = EditorState.create({
      doc: initialContent.current,
      extensions: [
        history(),
        lineNumbers(),
        highlightActiveLine(),
        highlightActiveLineGutter(),
        keymap.of([...defaultKeymap, ...historyKeymap]),
        StreamLanguage.define(toml),
        syntaxHighlighting(tomlHighlightStyle),
        theme,
        readOnlyComp.current.of(EditorState.readOnly.of(false)),
        EditorView.lineWrapping,
        EditorView.updateListener.of((update) => {
          if (update.docChanged && !isExternalUpdate.current) {
            onChangeRef.current(update.state.doc.toString());
          }
          isExternalUpdate.current = false;
        }),
      ],
    });

    const view = new EditorView({ state, parent: el });
    viewRef.current = view;

    return () => {
      view.destroy();
      viewRef.current = null;
    };
  }, []);

  // Toggle read-only without recreating the editor.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({
      effects: readOnlyComp.current.reconfigure(
        EditorState.readOnly.of(readOnly),
      ),
    });
  }, [readOnly]);

  // When content is replaced externally (e.g. config reload), update the doc.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const currentDoc = view.state.doc.toString();
    if (currentDoc === content) return;
    isExternalUpdate.current = true;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: content },
    });
  }, [content]);

  return <div ref={containerRef} className="h-full" />;
}

function ValidationPanel({ errors }: { errors: ValidationError[] }) {
  return (
    <div className="max-h-40 shrink-0 overflow-auto border-t border-border-default bg-surface px-4 py-2">
      <div className="eyebrow mb-1 text-danger">Validation errors</div>
      <ul className="space-y-0.5">
        {errors.map((e, i) => (
          <li key={i} className="font-mono text-xs text-danger">
            <span className="text-tertiary">
              L{e.line}:{e.column}
            </span>{" "}
            {e.message}
          </li>
        ))}
      </ul>
    </div>
  );
}

function HashMismatchDialog({
  serverHash,
  onReload,
  onDismiss,
}: {
  serverHash: string;
  onReload: () => void;
  onDismiss: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <Card header="Config changed on disk" className="max-w-md">
        <p className="text-sm text-secondary">
          The config file was modified since you last loaded it. The server's
          current hash is{" "}
          <code className="font-mono text-xs text-primary">
            {serverHash.slice(0, 12)}
          </code>
          . Reloading will discard your local changes.
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-md bg-elevated px-3 py-1.5 text-sm text-primary transition-colors hover:bg-border-strong"
          >
            Keep local changes
          </button>
          <button
            type="button"
            onClick={onReload}
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-[var(--color-base)] transition-colors hover:bg-accent/90"
          >
            Reload from server
          </button>
        </div>
      </Card>
    </div>
  );
}
