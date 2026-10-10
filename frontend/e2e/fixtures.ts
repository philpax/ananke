import type { Page, WebSocketRoute } from "@playwright/test";

import type {
  Bucket,
  Detail,
  Device,
  Schemas,
  Service,
} from "../src/api/contract";
import rows from "./reference-services.json" with { type: "json" };

const GiB = 1024 ** 3;
const bytes = (gib: number) => Math.round(gib * GiB);

export const referenceServices: Service[] = rows.map((row) => ({
  name: row.name,
  state: row.state,
  port: row.port,
  openai_compat: !row.embedding,
  lifecycle: "ondemand",
  priority: 80,
  pid: null,
  run_id: null,
  elastic_borrower: null,
  has_mmproj: row.vision,
  modality: row.embedding ? "embedding" : "chat",
  footprint_bytes: bytes(row.gib),
  footprint_devices: [
    { device: "cpu", bytes: bytes(Math.min(0.81, row.gib * 0.15)) },
    {
      device: "gpu:0",
      bytes: bytes((row.gib - Math.min(0.81, row.gib * 0.15)) / 2),
    },
    {
      device: "gpu:1",
      bytes: bytes((row.gib - Math.min(0.81, row.gib * 0.15)) / 2),
    },
  ],
  inflight_count: 0,
  last_used_ms: row.state === "running" ? Date.now() : null,
}));

export function hardware(count: number): Device[] {
  return [
    ...Array.from({ length: count }, (_, index) => ({
      id: `gpu:${index}`,
      name: "NVIDIA GeForce RTX 3090",
      total_bytes: 24 * GiB,
      free_bytes: bytes(index === 0 ? 24 - 19.9 * 0.55 : 24 - 18.75 * 0.92),
      reservations: [
        {
          service: "qwen3.6-27b",
          bytes: bytes(index === 0 ? 20.56 : 19.7),
          elastic: false,
        },
      ],
    })),
    {
      id: "cpu",
      name: "AMD Ryzen 9 7950X",
      total_bytes: bytes(125.67),
      free_bytes: bytes(125.67 - 45.93),
      reservations: [
        { service: "qwen3.6-27b", bytes: bytes(0.81), elastic: false },
      ],
    },
  ];
}

export function serviceDetail(name: string): Detail {
  const service =
    referenceServices.find((s) => s.name === name) ?? referenceServices[0];
  return {
    name,
    template: "llama-cpp",
    state: service.state,
    lifecycle: service.lifecycle,
    priority: service.priority,
    port: service.port,
    private_port: service.port + 30_000,
    placement_override: {},
    idle_timeout_ms: 600_000,
    pid: null,
    run_id: null,
    recent_logs: [],
    rolling_mean: null,
    rolling_samples: 0,
    rolling_mean_host: null,
    rolling_samples_host: 0,
    observed_peak_bytes: 0,
    elastic_borrower: null,
    model_info: null,
    estimate: {
      weights_bytes: bytes(17.25),
      kv_per_token: 0,
      configured_context: 204800,
      kv_bytes_for_context: bytes(16.43),
      compute_buffer_bytes_per_device: bytes(3.29),
    },
    placement_preview: {
      verdict: { kind: "fits" },
      expert_offload_bytes: 0,
      expert_offload_layers: 0,
      devices: (service.footprint_devices ?? []).map((p) => ({
        device: p.device,
        bytes: p.bytes,
        max_bytes: p.bytes,
        used_by_others_bytes: (p.device === "cpu" ? 45 : 10) * GiB,
        total_bytes: p.device === "cpu" ? bytes(125.67) : 24 * GiB,
      })),
    },
    current_allocation: {},
    modality: service.modality,
  };
}

export async function mockBackend(
  page: Page,
  count = 2,
  options: {
    readonly inventory?: readonly Service[];
    readonly events?: Schemas["Event"][];
  } = {},
) {
  let state = "running",
    content =
      '# Test configuration\n[daemon]\nmanagement_listen = "127.0.0.1:7071"\n',
    hash = "initial";
  const inventory = options.inventory ?? referenceServices;
  const events = [...(options.events ?? [])];
  const logPages = new Map<string, Schemas["LogsResponse"]>();
  let dropAction = false;
  const messages: Schemas["DashboardRequest"][] = [];
  const connections = new Map<
    WebSocketRoute,
    Map<string, Schemas["DashboardQuery"]>
  >();
  const calls = Object.assign(
    [] as {
      readonly method: string;
      readonly url: string;
      readonly body: string | null;
    }[],
    {
      control: {
        messages,
        logPage: (cursor: string, page: Schemas["LogsResponse"]) => {
          logPages.set(cursor, page);
        },
        dropNextAction: () => {
          dropAction = true;
        },
        connectionCount: 0,
        conflict: () => {
          hash = "external-change";
        },
        disconnect: () => {
          for (const socket of connections.keys())
            socket.close({ code: 1012, reason: "test reconnect" });
          connections.clear();
        },
        publish: (type: Schemas["DashboardQuery"]["type"], body: unknown) => {
          for (const [socket, queries] of connections)
            for (const [id, query] of queries)
              if (query.type === type) respond(socket, id, body);
        },
      },
    },
  );
  function respond(
    socket: WebSocketRoute,
    id: string,
    body: unknown,
    status = 200,
  ) {
    socket.send(JSON.stringify({ type: "response", id, status, body }));
  }
  function read(query: Schemas["DashboardQuery"]): unknown {
    switch (query.type) {
      case "services":
        return {
          services: inventory.map((s) =>
            s.name === "qwen3.6-27b" ? { ...s, state } : s,
          ),
          openai_api_port: 7070,
        };
      case "service":
        return {
          ...serviceDetail(query.name),
          state:
            query.name === "qwen3.6-27b"
              ? state
              : serviceDetail(query.name).state,
        };
      case "devices":
        return hardware(count);
      case "info":
        return {
          openai_listen: "127.0.0.1:7070",
          management_listen: "127.0.0.1:7071",
          uptime_ms: 6 * 86_400_000 + 4 * 3_600_000,
          config_path: "~/.config/ananke/config.toml",
        };
      case "config":
        return { content, hash, writable: true };
      case "events":
        return { since_ms: Date.now() - 60_000, events };
      case "models":
        return {
          object: "list",
          data: inventory
            .filter((s) => s.openai_compat && s.modality === "chat")
            .map((s) => ({
              id: s.name,
              object: "model",
              created: 0,
              owned_by: "ananke",
            })),
        };
      case "restarts":
        return { restarts: [] };
      case "samples":
        return { samples: [] };
      case "logs":
        if (query.before && logPages.has(query.before))
          return logPages.get(query.before);
        return {
          logs: [
            {
              timestamp_ms: Date.now() - 1000,
              stream: "stdout",
              line: "server is listening",
              run_id: 1,
              seq: 1,
            },
          ],
          next_cursor: null,
        };
      case "metrics": {
        const since = query.range.live_ms
          ? Date.now() - query.range.live_ms
          : query.range.since;
        const bucket: Bucket = {
          service: query.service ?? "qwen3.6-27b",
          bucket_start: Math.ceil(since / 300_000) * 300_000,
          request_count: 10,
          prompt_tokens: 100,
          completion_tokens: 200,
          avg_duration_ms: 4506,
          error_count: 1,
          avg_ttft_ms: 200,
          output_tps: 57.4,
          input_tps: 312.9,
          effective_tps: 44.3,
        };
        return { buckets: [bucket] };
      }
    }
  }
  function refresh() {
    for (const [socket, queries] of connections)
      for (const [id, query] of queries) respond(socket, id, read(query));
  }
  function readPath(query: Schemas["DashboardQuery"]) {
    let path =
      query.type === "service"
        ? `/api/services/${encodeURIComponent(query.name)}`
        : query.type === "logs"
          ? `/api/services/${encodeURIComponent(query.name)}/logs`
          : query.type === "models"
            ? "/v1/models"
            : query.type === "samples"
              ? "/api/devices/samples"
              : `/api/${query.type}`;
    if ("range" in query) {
      const until = query.range.live_ms ? Date.now() : query.range.until;
      const since = query.range.live_ms
        ? until - query.range.live_ms
        : query.range.since;
      const params = new URLSearchParams({
        since: String(since),
        until: String(until),
      });
      if ("service" in query && query.service)
        params.set("service", query.service);
      path += `?${params}`;
    }
    return path;
  }
  await page.routeWebSocket("**/api/dashboard", (socket) => {
    calls.control.connectionCount++;
    const queries = new Map<string, Schemas["DashboardQuery"]>();
    connections.set(socket, queries);
    socket.onClose(() => connections.delete(socket));
    socket.onMessage((frame) => {
      const request = JSON.parse(String(frame)) as Schemas["DashboardRequest"];
      messages.push(request);
      if (request.type === "cancel") {
        queries.delete(request.id);
        return;
      }
      if (request.type === "read") {
        calls.push({ method: "GET", url: readPath(request.query), body: null });
        if (request.subscribe) queries.set(request.id, request.query);
        respond(socket, request.id, read(request.query));
      } else if (request.type === "action") {
        calls.push({
          method: "POST",
          url: `/api/services/${request.name}/${request.action}`,
          body: null,
        });
        state =
          request.action === "disable"
            ? "disabled_user_disabled"
            : request.action === "stop" || request.action === "enable"
              ? "idle"
              : "running";
        if (dropAction) {
          dropAction = false;
          calls.control.disconnect();
        } else {
          respond(socket, request.id, null, 202);
          refresh();
        }
      } else if (request.type === "validate") {
        calls.push({
          method: "POST",
          url: "/api/config/validate",
          body: JSON.stringify({ content: request.content }),
        });
        respond(socket, request.id, {
          valid: !request.content.includes("invalid"),
          errors: request.content.includes("invalid")
            ? [{ line: 1, column: 1, message: "Invalid TOML." }]
            : [],
        });
      } else if (request.type === "save") {
        calls.push({
          method: "PUT",
          url: "/api/config",
          body: request.content,
        });
        if (request.hash !== hash)
          respond(socket, request.id, { error: "changed" }, 412);
        else {
          content = request.content;
          hash = "saved";
          respond(socket, request.id, null, 202);
          refresh();
        }
      } else if (request.type === "chat") {
        calls.push({
          method: "POST",
          url: "/v1/chat/completions",
          body: JSON.stringify(request.body),
        });
        const chunks: Schemas["ChatCompletionChunk"][] = [
          {
            choices: [
              {
                delta: {
                  reasoning_content: "Consider the request.\nThen answer.",
                },
              },
            ],
          },
          {
            choices: [
              { delta: { content: "Hello.\n\n```rust\nfn main() {}\n```" } },
            ],
          },
          {
            usage: { prompt_tokens: 12, completion_tokens: 5 },
            timings: { predicted_ms: 1000 },
          },
        ];
        for (const chunk of chunks)
          socket.send(
            JSON.stringify({
              type: "stream",
              id: request.id,
              text: `data: ${JSON.stringify(chunk)}\r\n\r\n`,
            }),
          );
        socket.send(
          JSON.stringify({
            type: "stream",
            id: request.id,
            text: "data: [DONE]\r\n\r\n",
          }),
        );
        socket.send(JSON.stringify({ type: "end", id: request.id }));
      }
    });
  });
  // Any accidental REST fallback fails the browser test immediately.
  await page.route(
    (url) =>
      url.pathname.startsWith("/api/") || url.pathname.startsWith("/v1/"),
    (route) => route.abort(),
  );
  return calls;
}
