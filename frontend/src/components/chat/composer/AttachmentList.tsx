// The files waiting to go with the next message, each removable.

import { useTranslation } from "react-i18next";

import { Icon } from "../../ui/icon/Icon.tsx";
import type { Attachment } from "../state/chatStore.ts";

type AttachmentListProps = {
  readonly attachments: readonly Attachment[];
  readonly onRemove: (index: number) => void;
};

export function AttachmentList({ attachments, onRemove }: AttachmentListProps) {
  const { t } = useTranslation();
  return (
    <ul className="flex flex-wrap gap-2 border-b border-line px-4 py-2">
      {attachments.map((att, i) => (
        <li
          key={`${i}-${att.name}`}
          className="flex max-w-full min-w-0 items-center gap-1.5 border border-line bg-raised py-0.5 pr-0.5 pl-1.5 text-xs text-body"
        >
          {att.type === "image" && (
            <img src={att.content} alt="" className="size-5 object-cover" />
          )}
          <span className="min-w-0 truncate">{att.name}</span>
          <button
            type="button"
            onClick={() => onRemove(i)}
            aria-label={t("chat.removeAttachment", { name: att.name })}
            title={t("chat.removeAttachment", { name: att.name })}
            className="inline-flex size-5 shrink-0 items-center justify-center text-muted hover:text-danger"
          >
            <Icon name="close" size="3" />
          </button>
        </li>
      ))}
    </ul>
  );
}
