// The markdown pipeline for replies and reasoning: GFM, KaTeX math, and
// highlight.js code (themed by `styles/highlight.css`). react-markdown
// escapes raw HTML by default, so no pre-sanitization is needed or
// wanted (see #24).

import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";

import { cn } from "../../ui/cn.ts";
import { CodeBlock } from "./CodeBlock.tsx";
import { codeBlockInfo } from "./codeBlock.ts";

export function Markdown({
  text,
  className,
}: {
  readonly text: string;
  readonly className?: string;
}) {
  return (
    <div
      className={cn(
        "min-w-0 text-md wrap-break-word text-ink",
        "[&_blockquote]:mb-3 [&_blockquote]:border-l-2 [&_blockquote]:border-model-accent [&_blockquote]:pl-3 [&_blockquote]:text-body",
        "[&_h1]:mb-3 [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:mb-3 [&_h2]:text-md [&_h2]:font-semibold [&_h3]:mb-2 [&_h3]:font-semibold",
        "[&_hr]:my-4 [&_hr]:border-line [&_li]:mb-1 [&_ol]:mb-3 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:mb-3 [&_ul]:mb-3 [&_ul]:list-disc [&_ul]:pl-5",
        "[&_td]:border [&_td]:border-line [&_td]:px-2 [&_td]:py-1 [&_th]:border [&_th]:border-line [&_th]:bg-raised [&_th]:px-2 [&_th]:py-1 [&_th]:text-left",
        "[&_:not(pre)>code]:bg-raised [&_:not(pre)>code]:px-1 [&_:not(pre)>code]:font-mono [&_:not(pre)>code]:text-sm",
        "[&_.katex-display]:overflow-x-auto [&_.katex-display]:overflow-y-hidden",
        "[&>*:last-child]:mb-0",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
        components={COMPONENTS}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

const COMPONENTS: Components = {
  // Fenced blocks arrive as <pre><code>, with or without a language;
  // inline code is a bare <code>, styled by the wrapper's selectors.
  pre: ({ node, children }) => (
    <CodeBlock info={codeBlockInfo(node)}>{children}</CodeBlock>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent-text underline hover:text-accent-hover"
    >
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="mb-3 overflow-x-auto">
      <table className="min-w-full border-collapse text-sm">{children}</table>
    </div>
  ),
};
