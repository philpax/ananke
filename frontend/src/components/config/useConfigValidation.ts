// Validates the draft against the daemon while it is being typed, once
// the typing pauses. A clean draft is the file on disk, so it is not
// checked and shows no errors.

import { useEffect } from "react";

import type { ValidationError } from "../../api/client.ts";
import { useValidateConfig } from "../../api/hooks.ts";

const VALIDATE_DEBOUNCE_MS = 600;

const NO_ERRORS: readonly ValidationError[] = [];

export function useConfigValidation(
  draft: string,
  dirty: boolean,
): readonly ValidationError[] {
  const validate = useValidateConfig();
  const { mutate } = validate;
  useEffect(() => {
    if (!dirty) return;
    const timer = setTimeout(() => mutate(draft), VALIDATE_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [draft, dirty, mutate]);
  if (!dirty || validate.data?.valid !== false) return NO_ERRORS;
  return validate.data.errors;
}
