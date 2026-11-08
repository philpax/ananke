// The bar under the transcript: the model's system prompt, the files
// waiting to go, the model picker with its copy and service links, and
// the message field with attach and send (or stop).

import { useTranslation } from "react-i18next";

import type { ServiceSummary } from "../../../api/client.ts";
import { Button } from "../../ui/button/Button.tsx";
import { ButtonLink } from "../../ui/button/ButtonLink.tsx";
import { buttonVariants } from "../../ui/button/buttonVariants.ts";
import { cn } from "../../ui/cn.ts";
import { CopyButton } from "../../ui/feedback/CopyButton.tsx";
import { Textarea } from "../../ui/form/Textarea.tsx";
import { Icon } from "../../ui/icon/Icon.tsx";
import {
  activeRun,
  addAttachment,
  cancel,
  removeAttachment,
  saveSystemPrompt,
  setInput,
  type ChatSnapshot,
} from "../state/chatStore.ts";
import { AttachmentList } from "./AttachmentList.tsx";
import { sendsOnEnter } from "./enterToSend.ts";
import { ModelDropdown } from "./ModelDropdown.tsx";
import { readAttachments } from "./readAttachments.ts";

type ComposerProps = {
  readonly chat: ChatSnapshot;
  readonly models: readonly ServiceSummary[];
  readonly onSelectModel: (name: string) => void;
  readonly onSend: () => void;
};

export function Composer({
  chat,
  models,
  onSelectModel,
  onSend,
}: ComposerProps) {
  const { t } = useTranslation();
  const model = chat.currentModel;
  const vision = models.find((s) => s.name === model)?.has_mmproj ?? false;
  const running = activeRun(chat.chatState) !== null;

  async function attach(files: FileList) {
    for (const att of await readAttachments(Array.from(files), vision)) {
      addAttachment(att);
    }
  }

  return (
    <div className="shrink-0 border-t border-line-strong bg-surface">
      {model && (
        <details className="border-b border-line px-4 py-2">
          <summary className="cursor-pointer text-sm font-semibold text-body select-none hover:text-ink">
            {t("chat.systemPrompt")}
          </summary>
          <Textarea
            mono
            value={chat.systemPrompt}
            aria-label={t("chat.systemPrompt")}
            placeholder={t("chat.systemPromptPlaceholder")}
            onChange={(e) => saveSystemPrompt(e.target.value)}
            className="mt-2 h-20 w-full resize-y text-sm"
          />
        </details>
      )}
      {chat.attachments.length > 0 && (
        <AttachmentList
          attachments={chat.attachments}
          onRemove={removeAttachment}
        />
      )}
      <div className="flex flex-col gap-2 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <ModelDropdown
            models={models}
            selected={model}
            onSelect={onSelectModel}
            className="min-w-0 flex-1 sm:max-w-md"
          />
          {model && (
            <>
              <CopyButton value={model} label={t("chat.modelName")} />
              <ButtonLink
                to={`/services/${encodeURIComponent(model)}`}
                size="icon-sm"
                tone="quiet"
                icon="open"
                aria-label={t("chat.openService", { model })}
                title={t("chat.openService", { model })}
              />
            </>
          )}
          <span className="text-xs text-muted max-sm:hidden pointer-coarse:hidden">
            {t("chat.enterHint")}
          </span>
        </div>
        <div className="flex items-stretch gap-2">
          <Textarea
            rows={2}
            value={chat.input}
            disabled={!model}
            aria-label={t("chat.message")}
            placeholder={
              model ? t("chat.typePlaceholder") : t("chat.selectModelFirst")
            }
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              const press = {
                key: e.key,
                shiftKey: e.shiftKey,
                isComposing: e.nativeEvent.isComposing,
                coarsePointer: coarsePointer(),
              };
              if (!sendsOnEnter(press)) return;
              e.preventDefault();
              if (!running) onSend();
            }}
            className="max-h-60 min-h-control-lg flex-1 resize-y max-sm:text-lg"
          />
          <label
            title={t("chat.attach")}
            className={cn(
              buttonVariants({ tone: "neutral", size: "icon" }),
              "h-auto cursor-pointer self-stretch focus-within:border-accent",
              !model && "pointer-events-none opacity-40",
            )}
          >
            <input
              type="file"
              multiple
              disabled={!model}
              aria-label={t("chat.attach")}
              className="sr-only"
              onChange={(e) => {
                if (e.target.files) void attach(e.target.files);
                e.target.value = "";
              }}
            />
            <Icon name="attach" />
          </label>
          {running ? (
            <Button
              tone="stop"
              icon="stop"
              onClick={cancel}
              className="h-auto self-stretch"
            >
              <span className="max-sm:sr-only">{t("chat.stop")}</span>
            </Button>
          ) : (
            <Button
              tone="primary"
              icon="send"
              onClick={onSend}
              disabled={!model || !chat.input.trim()}
              className="h-auto self-stretch"
            >
              <span className="max-sm:sr-only">{t("chat.send")}</span>
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function coarsePointer(): boolean {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches
  );
}
