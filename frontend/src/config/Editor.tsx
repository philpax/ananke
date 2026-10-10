import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import {
  HighlightStyle,
  StreamLanguage,
  syntaxHighlighting,
} from "@codemirror/language";
import { toml } from "@codemirror/legacy-modes/mode/toml";
import { Compartment, EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { tags } from "@lezer/highlight";
import { useEffect, useRef } from "preact/hooks";

export function Editor({
  value,
  editable,
  onChange,
}: {
  readonly value: string;
  readonly editable: boolean;
  readonly onChange: (text: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null),
    viewRef = useRef<EditorView | null>(null),
    callbackRef = useRef(onChange),
    initial = useRef(value),
    modeRef = useRef(new Compartment());
  useEffect(() => {
    callbackRef.current = onChange;
  }, [onChange]);
  useEffect(() => {
    if (!ref.current) return;
    const editor = new EditorView({
      parent: ref.current,
      state: EditorState.create({
        doc: initial.current,
        extensions: [
          history(),
          lineNumbers(),
          keymap.of([...defaultKeymap, ...historyKeymap]),
          StreamLanguage.define(toml),
          EditorView.lineWrapping,
          syntaxHighlighting(
            HighlightStyle.define([
              { tag: tags.comment, color: "var(--text-muted)" },
              { tag: tags.bracket, color: "var(--accent)" },
              { tag: tags.heading, color: "var(--accent)" },
            ]),
          ),
          modeRef.current.of([
            EditorState.readOnly.of(true),
            EditorView.editable.of(false),
          ]),
          EditorView.contentAttributes.of({
            "aria-label": "Configuration TOML",
          }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged)
              callbackRef.current(update.state.doc.toString());
          }),
          // CodeMirror owns its DOM; its theme API reaches descendants outside JSX.
          EditorView.theme(
            {
              "&": {
                backgroundColor: "var(--tone-surface)",
                color: "var(--text-body)",
                fontSize: "13px",
                lineHeight: "1.8",
              },
              ".cm-scroller": {
                fontFamily: "Iosevka,monospace",
                lineHeight: "1.8",
              },
              ".cm-content": { padding: "12px 0" },
              ".cm-line": { padding: "0 12px" },
              ".cm-gutters": {
                border: "none",
                backgroundColor: "var(--tone-surface)",
                color: "var(--text-faint)",
              },
              ".cm-lineNumbers .cm-gutterElement": {
                minWidth: "56px",
                padding: "0 2px 0 12px",
              },
              ".cm-cursor": { borderLeftColor: "var(--accent)" },
              "&.cm-focused .cm-selectionBackground": {
                backgroundColor: "var(--tone-selected)",
              },
            },
            { dark: true },
          ),
        ],
      }),
    });
    viewRef.current = editor;
    return () => {
      editor.destroy();
      viewRef.current = null;
    };
  }, []);
  useEffect(() => {
    const editor = viewRef.current;
    if (editor && editor.state.doc.toString() !== value)
      editor.dispatch({
        changes: { from: 0, to: editor.state.doc.length, insert: value },
      });
  }, [value]);
  useEffect(() => {
    const editor = viewRef.current;
    editor?.dispatch({
      effects: modeRef.current.reconfigure([
        EditorState.readOnly.of(!editable),
        EditorView.editable.of(editable),
      ]),
    });
    if (editable) editor?.focus();
  }, [editable]);
  return <div ref={ref} class="min-w-0 border border-line bg-surface" />;
}
