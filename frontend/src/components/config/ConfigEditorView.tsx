// Config editor (`/config`). A CodeMirror 6 TOML editor with debounced
// validation, ETag-gated save, reload, and a read-only toggle. Validation
// errors from the backend are shown in a panel below the editor with
// line/column references. A hash mismatch (412) surfaces a reload dialog
// so the operator can pull the server's current version.

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation, Trans } from "react-i18next";

import {
  useConfig,
  useValidateConfig,
  useSaveConfig,
} from "../../api/hooks.ts";
import { Button } from "../ui/Button.tsx";
import { Card } from "../ui/Card.tsx";
import { ViewHeader } from "../ui/ViewHeader.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { Badge } from "../ui/Badge.tsx";
import { CodeMirrorEditor } from "./CodeMirrorEditor.tsx";
import { ValidationPanel } from "./ValidationPanel.tsx";

const VALIDATE_DEBOUNCE_MS = 600;

export function ConfigEditorView() {
  const { t } = useTranslation();
  const config = useConfig();
  const validateMut = useValidateConfig();
  const saveMut = useSaveConfig();
  const [content, setContent] = useState("");
  const [originalHash, setOriginalHash] = useState("");
  const [readOnly, setReadOnly] = useState(false);
  const [hashMismatch, setHashMismatch] = useState<string | null>(null);
  const [loadedHash, setLoadedHash] = useState("");
  const [saveError, setSaveError] = useState<string | null>(null);

  const dirty =
    content !== "" && config.data && content !== config.data.content
      ? true
      : false;

  const serverReadOnly = config.data ? !config.data.writable : false;

  // Sync local state when server data changes (initial load, reload, save).
  // Using the "adjust state during render" pattern endorsed by React docs
  // instead of a useEffect, which would trigger cascading renders.
  const serverHash = config.data?.hash ?? "";
  if (serverHash !== "" && serverHash !== loadedHash) {
    setLoadedHash(serverHash);
    setContent(config.data?.content ?? "");
    setOriginalHash(serverHash);
    setHashMismatch(null);
    setReadOnly(serverReadOnly);
  }

  // Debounced validation: fire when content changes, after a short idle.
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!content || !dirty) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      validateMut.mutate(content);
    }, VALIDATE_DEBOUNCE_MS);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [content, dirty, validateMut]);

  // Clear save error when the user starts editing again.
  const handleChange = useCallback((value: string) => {
    setContent(value);
    setSaveError(null);
  }, []);

  const handleSave = useCallback(() => {
    if (!dirty || !originalHash) return;
    setSaveError(null);
    saveMut.mutate(
      { content, hash: originalHash },
      {
        onSuccess: (result) => {
          if (result.kind === "hash_mismatch") {
            setHashMismatch(result.serverHash);
          } else if (result.kind === "error") {
            setSaveError(result.message);
            // If the persist failed, switch to read-only so the
            // user doesn't keep trying to save.
            setReadOnly(true);
          }
        },
        onError: (err) => {
          setSaveError(err.message);
        },
      },
    );
  }, [content, dirty, originalHash, saveMut]);

  const handleReload = useCallback(() => {
    setHashMismatch(null);
    setSaveError(null);
    void config.refetch();
  }, [config]);

  const validationErrors =
    validateMut.data?.valid === false ? validateMut.data.errors : [];
  const saveErrors =
    saveMut.data?.kind === "validation_errors" ? saveMut.data.errors : [];
  const allErrors = [...validationErrors, ...saveErrors];

  if (config.isPending && !config.data) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  if (config.error) {
    return (
      <div className="p-4 text-sm text-danger">{config.error.message}</div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <ViewHeader>
        <h1 className="eyebrow !text-primary">{t("nav.config")}</h1>
        {readOnly && (
          <Badge variant="neutral">{t("config.readOnlyBadge")}</Badge>
        )}
        {dirty && <Badge variant="warning">{t("config.unsaved")}</Badge>}
        {saveMut.isPending && (
          <Badge variant="accent">{t("config.saving")}</Badge>
        )}
        <div className="ml-auto flex items-center gap-2">
          {!serverReadOnly && (
            <button
              type="button"
              onClick={() => setReadOnly((r) => !r)}
              className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                readOnly
                  ? "bg-elevated text-primary"
                  : "text-tertiary hover:text-secondary"
              }`}
            >
              {readOnly ? t("config.readOnly") : t("config.editable")}
            </button>
          )}
          <button
            type="button"
            onClick={handleReload}
            disabled={config.isFetching}
            className="rounded-md bg-elevated px-2.5 py-1 text-xs text-primary transition-colors hover:bg-border-strong disabled:opacity-40"
          >
            {t("config.reload")}
          </button>
          {!serverReadOnly && (
            <Button
              type="button"
              variant="iris"
              size="sm"
              onClick={handleSave}
              disabled={!dirty || saveMut.isPending || readOnly}
            >
              {t("config.save")}
            </Button>
          )}
        </div>
      </ViewHeader>

      {saveError && (
        <div className="shrink-0 border-b border-border-default bg-danger/10 px-4 py-2 text-xs text-danger">
          {saveError}
        </div>
      )}

      <div className="flex-1 overflow-hidden">
        <CodeMirrorEditor
          content={content}
          readOnly={readOnly}
          onChange={handleChange}
        />
      </div>

      {allErrors.length > 0 && <ValidationPanel errors={allErrors} />}

      {hashMismatch && (
        <HashMismatchDialog
          serverHash={hashMismatch}
          onReload={handleReload}
          onDismiss={() => setHashMismatch(null)}
        />
      )}
    </div>
  );
}

function HashMismatchDialog({
  serverHash,
  onReload,
  onDismiss,
}: {
  serverHash: string;
  onReload: () => void;
  onDismiss: () => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <Card header={t("config.changedOnDisk")} className="max-w-md">
        <p className="text-sm text-secondary">
          <Trans
            i18nKey="config.changedOnDiskBody"
            components={[<code className="font-mono text-xs text-primary" />]}
            values={{ hash: serverHash.slice(0, 12) }}
          />
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-md bg-elevated px-3 py-1.5 text-sm text-primary transition-colors hover:bg-border-strong"
          >
            {t("config.keepLocalChanges")}
          </button>
          <Button type="button" variant="iris" size="md" onClick={onReload}>
            {t("config.reloadFromServer")}
          </Button>
        </div>
      </Card>
    </div>
  );
}
