// The `/api/live` connection: one WebSocket per tab, shared by every live
// query and log follow. Subscriptions are refcounted by resource, so two
// components showing the services list hold one server subscription between
// them. A follow belongs to its holder, whose position it resumes from, so it
// is neither shared nor lingered. No React in here; `useLive.ts` and
// `useLogFollow.ts` adapt it.
//
// The server answers every subscribe with a full snapshot, so reconnecting
// is just resubscribing: nothing is replayed or patched. Each connection
// opens with a hello naming the daemon's build; a different build after a
// reconnect means the daemon was upgraded under this tab. A follow is
// re-sent with the position of the last line its holder received.

import { api } from "../client.ts";
import type {
  ApiError,
  LiveData,
  LiveFeed,
  LiveMessage,
  LiveRequest,
  LiveResource,
} from "../client.ts";
import type { FollowMessage } from "./logFollow.ts";

export type LiveStatus = "connecting" | "open" | "reconnecting";

export type LiveListener = {
  readonly onSnapshot: (data: LiveData) => void;
  /// A resource error, such as an unknown service. The subscription stays
  /// active; a later snapshot supersedes it.
  readonly onError: (error: ApiError, status: number) => void;
};

export type FollowListener = {
  /// The feed to send when the follow starts or the socket reopens.
  readonly resume: () => LiveFeed;
  readonly onMessage: (message: FollowMessage) => void;
};

export type LiveSocketOptions = {
  readonly url: () => string;
  /// Source of jitter in [0, 1); injectable so tests are deterministic.
  readonly random?: () => number;
};

/// How long a subscription outlives its last holder, so a route change that
/// unmounts one view and mounts another does not cancel and resubscribe.
export const LINGER_MS = 2_000;
export const BACKOFF_INITIAL_MS = 500;
export const BACKOFF_MAX_MS = 15_000;
const BACKOFF_JITTER = 0.2;

/// The reconnect delay after `attempt` consecutive failures: doubling from
/// 500 ms to a 15 s cap, spread by ±20 % so tabs do not reconnect in step.
export function backoffDelay(attempt: number, random: () => number): number {
  const base = Math.min(BACKOFF_MAX_MS, BACKOFF_INITIAL_MS * 2 ** attempt);
  return base * (1 + BACKOFF_JITTER * (2 * random() - 1));
}

/// A stable identity for a resource: subscriptions with equal keys share
/// one server subscription.
export function resourceKey(resource: LiveResource): string {
  switch (resource.kind) {
    case "services":
    case "devices":
    case "events":
    case "info":
      return resource.kind;
    case "service":
      return `service:${resource.name}`;
    default: {
      const unreachable: never = resource;
      return unreachable;
    }
  }
}

type Entry = {
  readonly id: string;
  readonly resource: LiveResource;
  readonly listeners: Set<LiveListener>;
  linger: ReturnType<typeof setTimeout> | null;
  last: LiveData | null;
};

export class LiveSocket {
  readonly #url: () => string;
  readonly #random: () => number;
  readonly #entries = new Map<string, Entry>();
  readonly #byId = new Map<string, Entry>();
  readonly #follows = new Map<string, FollowListener>();
  readonly #listeners = new Set<() => void>();
  #nextId = 0;
  #socket: WebSocket | null = null;
  #open = false;
  #status: LiveStatus = "connecting";
  #attempt = 0;
  #retry: ReturnType<typeof setTimeout> | null = null;
  #build: string | null = null;
  #daemonUpdated = false;

  constructor(options: LiveSocketOptions) {
    this.#url = options.url;
    this.#random = options.random ?? Math.random;
  }

  /// Hold a subscription to `resource` until the returned release is called.
  /// The listener gets the latest snapshot straight away if there is one.
  subscribe(resource: LiveResource, listener: LiveListener): () => void {
    const entry = this.#acquire(resource);
    entry.listeners.add(listener);
    if (entry.last !== null) listener.onSnapshot(entry.last);
    this.#ensureConnected();

    let released = false;
    return () => {
      if (released) return;
      released = true;
      entry.listeners.delete(listener);
      if (entry.listeners.size === 0) {
        entry.linger = setTimeout(() => this.#drop(entry), LINGER_MS);
      }
    };
  }

  /// Follow a feed until the returned release is called.
  follow(listener: FollowListener): () => void {
    const id = `f${this.#nextId++}`;
    this.#follows.set(id, listener);
    this.#send({ type: "follow", id, feed: listener.resume() });
    this.#ensureConnected();

    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.#follows.delete(id);
      this.#send({ type: "cancel", id });
    };
  }

  readonly getStatus = (): LiveStatus => this.#status;

  /// The build the first connection's hello named, or `null` before it.
  readonly getBuild = (): string | null => this.#build;

  /// Whether a reconnect reached a daemon with a different build than the
  /// one this tab first connected to.
  readonly getDaemonUpdated = (): boolean => this.#daemonUpdated;

  /// Notified when the status, `getBuild`, or `getDaemonUpdated` changes.
  readonly onChange = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  };

  /// Drop the current connection, if any, and connect again now.
  reconnect(): void {
    this.#clearRetry();
    this.#detach();
    this.#attempt = 0;
    this.#setStatus("reconnecting");
    this.#connect();
  }

  #acquire(resource: LiveResource): Entry {
    const key = resourceKey(resource);
    const existing = this.#entries.get(key);
    if (existing) {
      if (existing.linger !== null) {
        clearTimeout(existing.linger);
        existing.linger = null;
      }
      return existing;
    }
    const entry: Entry = {
      id: String(this.#nextId++),
      resource,
      listeners: new Set(),
      linger: null,
      last: null,
    };
    this.#entries.set(key, entry);
    this.#byId.set(entry.id, entry);
    this.#send({ type: "subscribe", id: entry.id, resource });
    return entry;
  }

  #drop(entry: Entry): void {
    this.#entries.delete(resourceKey(entry.resource));
    this.#byId.delete(entry.id);
    this.#send({ type: "cancel", id: entry.id });
  }

  #send(request: LiveRequest): void {
    // While closed, nothing is lost: opening resubscribes every entry.
    if (this.#open) this.#socket?.send(JSON.stringify(request));
  }

  #ensureConnected(): void {
    if (this.#socket === null && this.#retry === null) this.#connect();
  }

  #connect(): void {
    const socket = new WebSocket(this.#url());
    this.#socket = socket;
    socket.onopen = () => {
      this.#open = true;
      this.#attempt = 0;
      this.#setStatus("open");
      for (const entry of this.#entries.values()) {
        this.#send({
          type: "subscribe",
          id: entry.id,
          resource: entry.resource,
        });
      }
      for (const [id, listener] of this.#follows) {
        this.#send({ type: "follow", id, feed: listener.resume() });
      }
    };
    socket.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data !== "string") return;
      const message = parseMessage(ev.data);
      if (message !== null) this.#deliver(message);
    };
    socket.onerror = () => {
      // An error is not always followed promptly by a close; force one so
      // the reconnect cycle runs.
      socket.close();
    };
    socket.onclose = () => {
      if (this.#socket !== socket) return;
      this.#socket = null;
      this.#open = false;
      this.#setStatus("reconnecting");
      const delay = backoffDelay(this.#attempt, this.#random);
      this.#attempt += 1;
      this.#retry = setTimeout(() => {
        this.#retry = null;
        this.#connect();
      }, delay);
    };
  }

  #deliver(message: LiveMessage): void {
    switch (message.type) {
      case "hello": {
        if (this.#build === null) {
          this.#build = message.build;
          this.#notify();
        } else if (this.#build !== message.build && !this.#daemonUpdated) {
          this.#daemonUpdated = true;
          this.#notify();
        }
        return;
      }
      case "snapshot": {
        const entry = this.#byId.get(message.id);
        if (!entry) return;
        entry.last = message.data;
        for (const listener of [...entry.listeners]) {
          listener.onSnapshot(message.data);
        }
        return;
      }
      case "error": {
        const id = message.id ?? null;
        const follow = id === null ? undefined : this.#follows.get(id);
        if (follow) {
          follow.onMessage(message);
          return;
        }
        const entry = id === null ? undefined : this.#byId.get(id);
        if (!entry) {
          console.warn("live: the daemon rejected a request", message.error);
          return;
        }
        for (const listener of [...entry.listeners]) {
          listener.onError(message.error, message.status);
        }
        return;
      }
      case "log_lines":
      case "log_gap":
      case "log_caught_up":
        // A frame for a released follow is dropped.
        this.#follows.get(message.id)?.onMessage(message);
        return;
      default: {
        const unreachable: never = message;
        return unreachable;
      }
    }
  }

  #detach(): void {
    const socket = this.#socket;
    if (socket === null) return;
    this.#socket = null;
    this.#open = false;
    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;
    socket.close();
  }

  #clearRetry(): void {
    if (this.#retry !== null) {
      clearTimeout(this.#retry);
      this.#retry = null;
    }
  }

  #setStatus(status: LiveStatus): void {
    if (this.#status === status) return;
    this.#status = status;
    this.#notify();
  }

  #notify(): void {
    for (const listener of [...this.#listeners]) listener();
  }
}

/// The tab's shared connection. It opens on the first subscription.
export const liveSocket = new LiveSocket({ url: api.liveUrl });

// The daemon is the only producer, built from the same generated types, so
// the discriminant is checked and the rest is trusted.
function parseMessage(text: string): LiveMessage | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  return isLiveMessage(value) ? value : null;
}

function isLiveMessage(value: unknown): value is LiveMessage {
  if (typeof value !== "object" || value === null || !("type" in value)) {
    return false;
  }
  return (
    value.type === "hello" ||
    value.type === "snapshot" ||
    value.type === "log_lines" ||
    value.type === "log_gap" ||
    value.type === "log_caught_up" ||
    value.type === "error"
  );
}
