import type { ValidateFunction } from "ajv";

import type { Result } from "./client";
import { codec } from "./contract";
import type { Schemas } from "./contract";

type Request = Schemas["DashboardRequest"];
type Message = Schemas["DashboardMessage"];
type Command = Request extends infer R
  ? R extends { type: "action" | "save" | "validate" | "chat" }
    ? Omit<R, "id">
    : never
  : never;
export type Source<T> = {
  readonly query: Schemas["DashboardQuery"];
  readonly validate: ValidateFunction<T>;
};
type Pending = {
  readonly request: Request;
  readonly deliver: (message: Message) => void;
  readonly fail: (message: string) => void;
  sent: boolean;
};
const messageCodec = codec("DashboardMessage");
const validationCodec = codec("ConfigValidateResponse");
const subscriptions = new Map<
  string,
  { readonly request: Request; readonly deliver: (message: Message) => void }
>();
const pending = new Map<string, Pending>();
const listeners = new Set<(connected: boolean) => void>();
let socket: WebSocket | undefined;
let retry: ReturnType<typeof setTimeout> | undefined;
let sequence = 0;
let delay = 1000;

function connect() {
  if (socket || retry) return;
  const url = new URL("/api/dashboard", location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const current = new WebSocket(url);
  socket = current;
  current.onopen = () => {
    delay = 1000;
    for (const listener of listeners) listener(true);
    for (const { request } of subscriptions.values()) write(request);
    for (const operation of pending.values()) {
      write(operation.request);
      operation.sent = true;
    }
  };
  current.onmessage = (frame) => {
    let message: unknown;
    try {
      message = JSON.parse(frame.data);
    } catch {
      current.close(1002, "Invalid dashboard frame");
      return;
    }
    if (!messageCodec(message)) {
      current.close(1002, "Invalid dashboard response");
      return;
    }
    subscriptions.get(message.id)?.deliver(message);
    pending.get(message.id)?.deliver(message);
  };
  current.onclose = () => {
    if (socket !== current) return;
    socket = undefined;
    for (const listener of listeners) listener(false);
    for (const [id, operation] of pending) {
      if (operation.sent) {
        pending.delete(id);
        operation.fail(
          "Dashboard: the connection closed before acknowledgement. Check the current state before retrying.",
        );
      }
    }
    // Reconnect restores read subscriptions; commands are never replayed.
    retry = setTimeout(() => {
      retry = undefined;
      connect();
    }, delay);
    delay = Math.min(delay * 2, 30_000);
  };
}

function write(request: Request) {
  if (socket?.readyState === WebSocket.OPEN)
    socket.send(JSON.stringify(request));
}

export function connectionStatus(listener: (connected: boolean) => void) {
  listeners.add(listener);
  listener(socket?.readyState === WebSocket.OPEN);
  connect();
  return () => {
    listeners.delete(listener);
  };
}

export function watch<T>(
  source: Source<T>,
  deliver: (result: Result<T>) => void,
) {
  const id = String(++sequence);
  const request: Request = {
    type: "read",
    id,
    query: source.query,
    subscribe: true,
  };
  subscriptions.set(id, {
    request,
    deliver: (message) => {
      if (message.type !== "response") return;
      if (message.status < 200 || message.status >= 300)
        deliver({ kind: "error", message: responseError(message) });
      else if (source.validate(message.body))
        deliver({ kind: "ok", value: message.body });
      else
        deliver({
          kind: "error",
          message: "Dashboard: the server returns an invalid resource.",
        });
    },
  });
  connect();
  write(request);
  return () => {
    subscriptions.delete(id);
    write({ type: "cancel", id });
  };
}

export async function read<T>(source: Source<T>, signal?: AbortSignal) {
  const result = await operate(
    { type: "read", query: source.query, subscribe: false },
    signal,
  );
  if (result.kind === "error") return result;
  return source.validate(result.value)
    ? { kind: "ok" as const, value: result.value }
    : {
        kind: "error" as const,
        message: "Dashboard: the server returns an invalid resource.",
      };
}

export function command(
  request: Command,
  signal?: AbortSignal,
  onStream?: (text: string) => void,
): Promise<Result<unknown>> {
  return operate(request, signal, onStream);
}

function operate(
  operation: Command | Omit<Extract<Request, { type: "read" }>, "id">,
  signal?: AbortSignal,
  onStream?: (text: string) => void,
): Promise<Result<unknown>> {
  if (signal?.aborted)
    return Promise.resolve({ kind: "error", message: "Request cancelled." });
  if (operation.type !== "read" && socket?.readyState !== WebSocket.OPEN)
    return Promise.resolve({
      kind: "error",
      message: "Dashboard: reconnecting. Retry when the connection returns.",
    });
  const id = String(++sequence);
  const request: Request = { ...operation, id };
  return new Promise((resolve) => {
    function finish(result: Result<unknown>) {
      pending.delete(id);
      signal?.removeEventListener("abort", abort);
      resolve(result);
    }
    function abort() {
      write({ type: "cancel", id });
      finish({ kind: "error", message: "Request cancelled." });
    }
    pending.set(id, {
      request,
      sent: socket?.readyState === WebSocket.OPEN,
      fail: (message) => finish({ kind: "error", message }),
      deliver: (message) => {
        if (message.type === "stream") {
          try {
            onStream?.(message.text);
          } catch (error) {
            write({ type: "cancel", id });
            finish({
              kind: "error",
              message: error instanceof Error ? error.message : String(error),
            });
          }
        } else if (message.type === "end")
          finish({ kind: "ok", value: undefined });
        else
          finish(
            message.status >= 200 && message.status < 300
              ? { kind: "ok", value: message.body }
              : { kind: "error", message: responseError(message) },
          );
      },
    });
    signal?.addEventListener("abort", abort, { once: true });
    connect();
    write(request);
  });
}

function responseError(message: Extract<Message, { type: "response" }>) {
  if (message.status === 412)
    return "Config: the file changed on the server. Reload it and review your edits before saving.";
  const body = message.body;
  if (validationCodec(body) && !body.valid)
    return body.errors
      .map(
        (error) =>
          `Line ${error.line}, column ${error.column}: ${error.message}`,
      )
      .join("\n");
  if (body && typeof body === "object" && "error" in body) {
    const error = body.error;
    if (typeof error === "string") return error;
    if (
      error &&
      typeof error === "object" &&
      "message" in error &&
      typeof error.message === "string"
    )
      return error.message;
  }
  return `Dashboard: request failed (${message.status}).`;
}
