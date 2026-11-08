// Chat with a model on the OpenAI-compatible API, a web equivalent of
// `anankectl chat`: a model picker, a per-model system prompt, streamed
// replies with their reasoning and token figures, and file attachments.
// A stopped model is started first, and the chat waits for it.
//
// The conversation lives in the module-level chat store, so it survives
// navigating away and back within the tab.

import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router-dom";

import { isChatModel } from "../../api/chatModels.ts";
import { useInfo, useServices } from "../../api/hooks.ts";
import { openaiBaseUrlFromListen } from "../../util.ts";
import { Button } from "../ui/button/Button.tsx";
import { Spinner } from "../ui/feedback/Spinner.tsx";
import { PageHeader } from "../ui/layout/PageHeader.tsx";
import { conversationMeta } from "./chatMeta.ts";
import { Composer } from "./composer/Composer.tsx";
import {
  clearConversation,
  selectModel,
  send,
  useChat,
} from "./state/chatStore.ts";
import { Transcript } from "./transcript/Transcript.tsx";

export function ChatView() {
  const { t } = useTranslation();
  const services = useServices();
  const info = useInfo();
  const [searchParams, setSearchParams] = useSearchParams();
  const chat = useChat();
  const paramModel = searchParams.get("model");

  // The store owns the selection, so it survives navigating away and
  // back; the URL is how a deep link or Back names a model. A model the
  // URL names is applied when the URL changes, and the dropdown writes
  // both, so neither can hold a model the other has left.
  useEffect(() => {
    if (paramModel !== null) selectModel(paramModel);
  }, [paramModel]);
  useEffect(() => {
    if (paramModel === null && chat.currentModel !== null) {
      setSearchParams({ model: chat.currentModel }, { replace: true });
    }
  }, [paramModel, chat.currentModel, setSearchParams]);

  if (services.isPending) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  const models = (services.data ?? [])
    .filter(isChatModel)
    .sort(
      (a, b) =>
        Number(b.state === "running") - Number(a.state === "running") ||
        a.name.localeCompare(b.name),
    );
  const model = chat.currentModel;
  const meta = conversationMeta(chat.messages);

  function handleSelectModel(name: string) {
    selectModel(name);
    setSearchParams({ model: name });
  }

  function handleSend() {
    if (!model) return;
    void send(model, openaiBaseUrlFromListen(info.data?.openai_listen));
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PageHeader
        title={t("chat.title")}
        meta={
          <>
            <span className="font-mono text-body">
              {model ?? t("chat.noModel")}
            </span>
            {" · "}
            {t("chat.exchanges", { count: meta.exchanges })}
            {meta.contextTokens !== null && (
              <>
                {" · "}
                {t("chat.contextTokens", { count: meta.contextTokens })}
              </>
            )}
          </>
        }
        actions={
          <Button
            icon="plus"
            onClick={clearConversation}
            disabled={
              chat.messages.length === 0 && chat.chatState.kind === "idle"
            }
          >
            {t("chat.newChat")}
          </Button>
        }
      />
      <Transcript
        messages={chat.messages}
        chatState={chat.chatState}
        liveStats={chat.stats}
      />
      <Composer
        chat={chat}
        models={models}
        onSelectModel={handleSelectModel}
        onSend={handleSend}
      />
    </div>
  );
}
