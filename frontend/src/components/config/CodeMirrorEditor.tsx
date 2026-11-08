// The CodeMirror 6 TOML editor for the Config view, themed from the design
// tokens, with line numbers, the active line marked, and long lines
// wrapped. It holds its own document; `content` replaces it only when the
// two differ, so typing never round-trips through React.

import { useEffect, useRef } from "react";
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

// TOML highlighting from the shared `syntax-*` tokens, so the editor
// matches highlighted code elsewhere. The browser resolves the `var()`s
// live, so the editor follows a theme change without a rebuild.
const tomlHighlightStyle = HighlightStyle.define([
  {
    tag: tags.comment,
    color: "var(--color-syntax-comment)",
    fontStyle: "italic",
  },
  { tag: tags.string, color: "var(--color-syntax-string)" },
  { tag: tags.number, color: "var(--color-syntax-number)" },
  { tag: tags.bool, color: "var(--color-syntax-keyword)" },
  { tag: tags.keyword, color: "var(--color-syntax-keyword)" },
  { tag: tags.propertyName, color: "var(--color-syntax-property)" },
  {
    tag: tags.definition(tags.propertyName),
    color: "var(--color-syntax-property)",
  },
  { tag: tags.variableName, color: "var(--color-ink)" },
  { tag: tags.atom, color: "var(--color-syntax-keyword)" },
  { tag: tags.punctuation, color: "var(--color-syntax-punct)" },
  { tag: tags.bracket, color: "var(--color-syntax-punct)" },
]);

export function CodeMirrorEditor({
  content,
  readOnly,
  label,
  onChange,
}: {
  readonly content: string;
  readonly readOnly: boolean;
  readonly label: string;
  readonly onChange: (value: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const readOnlyComp = useRef(new Compartment());
  const onChangeRef = useRef(onChange);
  const isExternalUpdate = useRef(false);
  const initialContent = useRef(content);
  const initialLabel = useRef(label);

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
        color: "var(--color-ink)",
        height: "100%",
        fontSize: "13px",
      },
      ".cm-scroller": {
        fontFamily: "var(--font-mono)",
        overflow: "auto",
      },
      ".cm-content": { padding: "8px 0" },
      ".cm-gutters": {
        backgroundColor: "var(--color-surface)",
        color: "var(--color-muted)",
        border: "none",
        borderRight: "1px solid var(--color-line)",
      },
      ".cm-lineNumbers .cm-gutterElement": {
        fontFamily: "var(--font-mono)",
        fontSize: "11px",
        padding: "0 8px 0 12px",
      },
      ".cm-activeLine": { backgroundColor: "var(--color-raised)" },
      ".cm-activeLineGutter": {
        backgroundColor: "var(--color-raised)",
        color: "var(--color-soft)",
      },
      "&.cm-focused .cm-selectionBackground, ::selection": {
        backgroundColor: "var(--color-selected)",
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
        EditorView.contentAttributes.of({
          "aria-label": initialLabel.current,
        }),
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
