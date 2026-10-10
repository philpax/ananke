import DOMPurify from "dompurify";
import { marked } from "marked";
import { useEffect, useRef } from "preact/hooks";

import { Copy } from "../ui/Copy";

export function Markdown({ text }: { readonly text: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const tokens = marked.lexer(text);
  const rendered = tokens.map((token, index) =>
    token.type === "code" ? (
      <div key={index} class="mb-3 border border-line bg-canvas">
        <div
          class="
            flex items-center justify-between gap-2.5 border-b border-line
            bg-raised px-2 py-1.5
          "
        >
          <span class="font-mono text-xs text-muted">
            {token.lang || "text"}
          </span>
          <Copy text={token.text} label="code" />
        </div>
        <pre
          class="
            m-0 overflow-x-auto p-3 font-mono text-[12.5px] leading-[1.7]
            text-body
          "
        >
          <code>{token.text}</code>
        </pre>
      </div>
    ) : (
      <div
        key={index}
        dangerouslySetInnerHTML={{
          __html: DOMPurify.sanitize(marked.parser([token]), {
            FORBID_TAGS: ["img", "style", "form", "input", "iframe"],
          }),
        }}
      />
    ),
  );
  useEffect(() => {
    for (const a of ref.current?.querySelectorAll("a") ?? []) {
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener noreferrer");
    }
  }, [text]);
  return (
    <div
      ref={ref}
      class="
        text-[14.5px] leading-[1.65] text-ink
        [&_blockquote]:border-l-2 [&_blockquote]:border-model-accent
        [&_blockquote]:pl-3
        [&_code]:font-mono [&_code]:text-[12.5px]
        [&_h1]:mb-3 [&_h1]:text-lg
        [&_h2]:mb-3 [&_h2]:text-base
        [&_h3]:mb-3 [&_h3]:font-semibold
        [&_ol]:mb-3 [&_ol]:list-decimal [&_ol]:pl-5
        [&_p]:mb-3
        [&_table]:mb-3 [&_table]:w-full
        [&_td]:border [&_td]:border-line [&_td]:p-2
        [&_th]:border [&_th]:border-line [&_th]:p-2
        [&_ul]:mb-3 [&_ul]:list-disc [&_ul]:pl-5
      "
    >
      {rendered}
    </div>
  );
}
