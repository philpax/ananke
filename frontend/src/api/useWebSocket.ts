// Reusable WebSocket hook. Encapsulates the connection lifecycle that
// was duplicated between the events WebSocket (events.ts) and the log
// stream WebSocket (hooks.ts): connect with exponential-ish reconnect,
// onerror forces close to trigger onclose, shouldReconnect prevents
// zombie reconnections after cleanup, and a reconnect() function allows
// forced reconnection.
//
// Handlers are stored in a ref so they can change on every render
// without causing the WebSocket to reconnect. The effect only re-runs
// when the URL changes.

import { useCallback, useEffect, useRef, useState } from "react";

type WebSocketHandlers = {
  onOpen?: () => void;
  onClose?: () => void;
  onMessage: (data: string) => void;
};

export function useWebSocket(
  url: string | null,
  handlers: WebSocketHandlers,
  reconnectDelayMs: number = 2_000,
): { connected: boolean; reconnect: () => void } {
  const [connected, setConnected] = useState(false);
  const handlersRef = useRef(handlers);
  useEffect(() => {
    handlersRef.current = handlers;
  });
  const reconnectRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (url === null) return;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let shouldReconnect = true;

    const connect = () => {
      if (!shouldReconnect) return;
      socket = new WebSocket(url);
      socket.onopen = () => {
        setConnected(true);
        handlersRef.current.onOpen?.();
      };
      socket.onclose = () => {
        setConnected(false);
        handlersRef.current.onClose?.();
        if (!shouldReconnect) return;
        reconnectTimer = window.setTimeout(connect, reconnectDelayMs);
      };
      socket.onerror = () => {
        // onerror doesn't always trigger onclose promptly, so close
        // the socket explicitly to force the reconnect cycle.
        socket?.close();
      };
      socket.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        handlersRef.current.onMessage(ev.data);
      };
    };
    connect();

    reconnectRef.current = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (socket) {
        socket.onclose = null;
        socket.onerror = null;
        socket.onmessage = null;
        socket.close();
        socket = null;
      }
      connect();
    };

    return () => {
      shouldReconnect = false;
      reconnectRef.current = null;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [url, reconnectDelayMs]);

  const reconnect = useCallback(() => {
    reconnectRef.current?.();
  }, []);

  return { connected, reconnect };
}
