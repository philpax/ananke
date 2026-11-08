// Config (`/config`): the daemon's TOML in an editor, validated while it
// is typed and saved against the hash it was read at. The draft lives in a
// store outside the view (`configDraftStore`), so leaving the view keeps
// unsaved edits, and a newer version on disk never replaces them silently:
// a reload event or a refused save holds it for the operator to take or
// set aside.

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { useConfig, useSaveConfig } from "../../api/hooks.ts";
import { Button } from "../ui/button/Button.tsx";
import { Message } from "../ui/feedback/Message.tsx";
import { PageHeader } from "../ui/layout/PageHeader.tsx";
import { Chip } from "../ui/status/Chip.tsx";
import { CodeMirrorEditor } from "./CodeMirrorEditor.tsx";
import {
  DiskChangeNotice,
  ReadOnlyNotice,
  SaveErrorNotice,
} from "./ConfigNotices.tsx";
import { ConflictDialog } from "./ConflictDialog.tsx";
import { ValidationPanel } from "./ValidationPanel.tsx";
import {
  conflict,
  discard,
  edit,
  isDirty,
  keepDraft,
  receive,
  saved,
} from "./configDraft.ts";
import { updateDraft, useDraft } from "./configDraftStore.ts";
import { useConfigValidation } from "./useConfigValidation.ts";

export function ConfigEditorView() {
  const { t } = useTranslation();
  const config = useConfig();
  const save = useSaveConfig();
  const [locked, setLocked] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // `receive` is idempotent, so the fetched copy is applied while
  // rendering (no frame of stale state) and committed after.
  const server = config.data;
  const readAt = config.dataUpdatedAt;
  const stored = useDraft();
  const state = server ? receive(stored, server, readAt) : stored;
  useEffect(() => {
    if (server) updateDraft((s) => receive(s, server, readAt));
  }, [server, readAt]);

  const { base, draft, diskChange } = state;
  const dirty = isDirty(state);
  const validationErrors = useConfigValidation(draft, dirty);
  const saveErrors =
    save.data?.kind === "validation_errors" && save.variables?.content === draft
      ? save.data.errors
      : [];
  const errors = saveErrors.length > 0 ? saveErrors : validationErrors;

  if (base === null) {
    return (
      <div className="flex h-full flex-col">
        <PageHeader title={t("config.title")} />
        <div className="px-4">
          {config.error ? (
            <Message kind="error" text={config.error.message} />
          ) : (
            <Message kind="loading" />
          )}
        </div>
      </div>
    );
  }

  const writable = base.writable;
  const readOnly = !writable || locked;

  function handleSave() {
    if (base === null || !dirty) return;
    setSaveError(null);
    const content = draft;
    save.mutate(
      { content, hash: base.hash },
      {
        onSuccess: (result) => {
          switch (result.kind) {
            case "ok":
              updateDraft((s) => saved(s, content));
              break;
            case "hash_mismatch":
              updateDraft((s) => conflict(s, result.serverHash));
              break;
            case "validation_errors":
              break;
            case "error":
              setSaveError(result.message);
              // The file could not be written; stop offering a save that
              // will fail the same way.
              setLocked(true);
              break;
            default: {
              const unreachable: never = result;
              return unreachable;
            }
          }
        },
        onError: (err) => setSaveError(err.message),
      },
    );
  }

  function handleReload() {
    setSaveError(null);
    updateDraft(discard);
    void config.refetch();
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PageHeader
        title={t("config.title")}
        meta={
          <span className="font-mono text-xs text-soft">
            {base.path}
            {state.readAtMs !== null &&
              ` · ${t("config.readAt", {
                time: new Date(state.readAtMs).toLocaleTimeString(),
              })}`}
          </span>
        }
        actions={
          <>
            <span role="status" className="flex gap-1.5">
              {!writable && <Chip>{t("config.readOnlyBadge")}</Chip>}
              {dirty && <Chip tone="warn">{t("config.unsaved")}</Chip>}
              {save.isPending && (
                <Chip tone="accent">{t("config.saving")}</Chip>
              )}
              {state.saved && !dirty && !save.isPending && (
                <Chip tone="ok">{t("config.saved")}</Chip>
              )}
            </span>
            {writable && (
              <Button
                size="sm"
                tone="quiet"
                aria-pressed={locked}
                onClick={() => setLocked((l) => !l)}
              >
                {t("config.lockEditing")}
              </Button>
            )}
            <Button
              size="sm"
              icon="restart"
              title={dirty ? t("config.discardTitle") : undefined}
              disabled={config.isFetching}
              onClick={handleReload}
            >
              {dirty ? t("config.discard") : t("config.reload")}
            </Button>
            {writable && (
              <Button
                size="sm"
                tone="primary"
                disabled={!dirty || save.isPending || readOnly}
                onClick={handleSave}
              >
                {t("config.save")}
              </Button>
            )}
          </>
        }
      />
      {!writable && <ReadOnlyNotice />}
      {diskChange?.cause === "reload" && (
        <DiskChangeNotice
          onReload={() => updateDraft(discard)}
          onKeep={() => updateDraft(keepDraft)}
        />
      )}
      {saveError && <SaveErrorNotice message={saveError} />}
      <div className="min-h-0 flex-1">
        <CodeMirrorEditor
          content={draft}
          readOnly={readOnly}
          label={t("config.editorLabel")}
          onChange={(value) => {
            setSaveError(null);
            updateDraft((s) => edit(s, value));
          }}
        />
      </div>
      {errors.length > 0 && <ValidationPanel errors={errors} />}
      {diskChange?.cause === "conflict" && (
        <ConflictDialog
          serverHash={diskChange.hash}
          onKeep={() => updateDraft(keepDraft)}
          onReload={() => updateDraft(discard)}
        />
      )}
    </div>
  );
}
