// Which services can be chatted with. A service needs to be on the
// OpenAI-compatible API at all (a command service without an
// `openai_proxy` block is not, whatever its modality) and to serve chat
// rather than embeddings.

import type { ServiceSummary } from "./client.ts";

export type ChatCapable = Pick<ServiceSummary, "openai_compat" | "modality">;

export function isChatModel(service: ChatCapable): boolean {
  return (
    service.openai_compat &&
    (service.modality === undefined || service.modality === "chat")
  );
}
