// A scriptable stand-in for the browser WebSocket, for the live socket's
// tests. Install it with `vi.stubGlobal("WebSocket", FakeWebSocket)`.

import type { LiveMessage, LiveRequest } from "../client.ts";

export class FakeWebSocket {
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  readonly sent: LiveRequest[] = [];
  closed = false;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  static latest(): FakeWebSocket {
    const socket = FakeWebSocket.instances.at(-1);
    if (!socket) throw new Error("no socket was opened");
    return socket;
  }

  static reset(): void {
    FakeWebSocket.instances = [];
  }

  send(text: string): void {
    this.sent.push(JSON.parse(text));
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.onclose?.();
  }

  /// Server side: accept the handshake.
  accept(): void {
    this.onopen?.();
  }

  /// Server side: deliver a frame.
  receive(message: LiveMessage): void {
    this.onmessage?.({ data: JSON.stringify(message) });
  }
}
