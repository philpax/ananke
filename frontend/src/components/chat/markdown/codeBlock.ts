// What a fenced code block's header shows and its copy button copies,
// read from the markdown tree rather than the rendered DOM, so the copy
// is the source text and not rehype-highlight's spans.

import type { ExtraProps } from "react-markdown";

type HastElement = NonNullable<ExtraProps["node"]>;
type HastNode = HastElement["children"][number];

export type CodeBlockInfo = {
  /// The fence's language, or `null` for a bare fence.
  readonly language: string | null;
  readonly text: string;
};

/// Reads a `<pre>` element whose child is the `<code>` of a fenced block.
export function codeBlockInfo(pre: HastElement | undefined): CodeBlockInfo {
  const code = pre?.children.find(
    (child): child is HastElement =>
      child.type === "element" && child.tagName === "code",
  );
  if (!code) return { language: null, text: pre ? textOf(pre) : "" };
  const classes = code.properties.className;
  const language = Array.isArray(classes)
    ? classes
        .map(String)
        .find((c) => c.startsWith("language-"))
        ?.slice("language-".length)
    : undefined;
  return {
    language: language || null,
    text: textOf(code).replace(/\n$/, ""),
  };
}

function textOf(node: HastNode | HastElement): string {
  if (node.type === "text") return node.value;
  if (node.type === "element") return node.children.map(textOf).join("");
  return "";
}
