import { useEffect, useRef, useState } from "preact/hooks";

import { streamChat } from "./stream";
import type { ChatTurn } from "./stream";

export function useChat() {
  const [turns, setTurns] = useState<readonly ChatTurn[]>([]),
    [model, setModel] = useState(""),
    [system, setSystem] = useState(""),
    [input, setInput] = useState(""),
    [error, setError] = useState<string>(),
    [generating, setGenerating] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => controllerRef.current?.abort(), []);
  async function send() {
    if (controllerRef.current || !input.trim() || !model) return;
    const user: ChatTurn = {
      role: "user",
      content: input,
      reasoning: "",
      model,
      at: Date.now(),
    };
    const history = [...turns, user];
    let reply: ChatTurn = {
      role: "assistant",
      content: "",
      reasoning: "",
      model,
      at: Date.now(),
    };
    setTurns([...history, reply]);
    setInput("");
    setError(undefined);
    setGenerating(true);
    const run = new AbortController();
    controllerRef.current = run;
    let decodeStarted: number | undefined;
    let decodedMilliseconds: number | undefined;
    const result = await streamChat(
      { model, system, turns: history },
      run.signal,
      (chunk) => {
        const delta = chunk.choices?.[0]?.delta;
        const content = delta?.content ?? "",
          reasoning = delta?.reasoning_content ?? delta?.reasoning ?? "";
        if (content && decodeStarted === undefined)
          decodeStarted = performance.now();
        if (chunk.timings?.predicted_ms != null)
          decodedMilliseconds = chunk.timings.predicted_ms;
        reply = {
          ...reply,
          content: reply.content + content,
          reasoning: reply.reasoning + reasoning,
          inputTokens: chunk.usage?.prompt_tokens ?? reply.inputTokens,
          outputTokens: chunk.usage?.completion_tokens ?? reply.outputTokens,
        };
        if (reply.outputTokens != null) {
          const duration =
            decodedMilliseconds ??
            (decodeStarted === undefined
              ? 0
              : performance.now() - decodeStarted);
          reply = {
            ...reply,
            speed:
              duration > 0 ? (reply.outputTokens / duration) * 1000 : undefined,
          };
        }
        setTurns([...history, reply]);
      },
    );
    if (controllerRef.current !== run) return;
    controllerRef.current = null;
    setGenerating(false);
    if (result.kind === "error") setError(result.message);
  }
  return {
    turns,
    model,
    setModel,
    system,
    setSystem,
    input,
    setInput,
    error,
    generating,
    send,
    stop: () => controllerRef.current?.abort(),
    clear: () => {
      controllerRef.current?.abort();
      controllerRef.current = null;
      setTurns([]);
      setError(undefined);
      setGenerating(false);
    },
  };
}
