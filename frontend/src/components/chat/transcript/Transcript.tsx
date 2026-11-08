// The conversation's scroll region. It follows new content while the
// reader is at the bottom, shows the wait between a message and its
// reply (with the model's start, which can take minutes), and the last
// error.

import { useTranslation } from "react-i18next";

import { useStickToBottom } from "../../../hooks/useStickToBottom.ts";
import { EmptyState } from "../../ui/feedback/EmptyState.tsx";
import { Message } from "../../ui/feedback/Message.tsx";
import { Spinner } from "../../ui/feedback/Spinner.tsx";
import {
  activeRun,
  type ChatState,
  type Message as ChatMessage,
  type StreamStats,
} from "../state/chatStore.ts";
import { Turn } from "./Turn.tsx";

type TranscriptProps = {
  readonly messages: readonly ChatMessage[];
  readonly chatState: ChatState;
  readonly liveStats: StreamStats;
};

export function Transcript({
  messages,
  chatState,
  liveStats,
}: TranscriptProps) {
  const { t } = useTranslation();
  const { scrollRef, onScroll } = useStickToBottom(messages);
  const run = activeRun(chatState);
  const last = messages[messages.length - 1];
  const awaitingReply = run !== null && last === run.turn;

  return (
    <div
      ref={scrollRef}
      onScroll={onScroll}
      className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pt-4 pb-6"
    >
      {messages.length === 0 && (
        <EmptyState
          icon="chat"
          message={t("chat.emptyState")}
          className="flex-1"
        />
      )}
      {messages.map((message) => (
        <Turn
          key={message.id}
          message={message}
          liveStats={
            chatState.kind === "streaming" && message === last
              ? liveStats
              : null
          }
        />
      ))}
      {awaitingReply && (
        <div role="status" className="flex items-start gap-2 text-sm">
          <Spinner className="mt-0.5" />
          <div>
            <p className="text-body">
              {chatState.kind === "starting"
                ? t("chat.starting", { model: run.model })
                : t("chat.waiting", { model: run.model })}
            </p>
            {chatState.kind === "starting" && (
              <p className="text-muted">{t("chat.startingHint")}</p>
            )}
          </div>
        </div>
      )}
      {chatState.kind === "error" && (
        <Message kind="error" text={chatState.message} />
      )}
    </div>
  );
}
