import { describe, expect, it } from "vitest";

import type { ConfigResponse } from "../../api/client.ts";
import {
  INITIAL_DRAFT,
  conflict,
  discard,
  edit,
  isDirty,
  keepDraft,
  receive,
  saved,
  type DraftState,
} from "./configDraft.ts";

function server(content: string, hash = `h:${content}`): ConfigResponse {
  return { content, hash, path: "/etc/ananke.toml", writable: true };
}

const loaded = receive(INITIAL_DRAFT, server("a = 1"), 100);
const editing = edit(loaded, "a = 2");

describe("receive", () => {
  it("adopts the first version it sees", () => {
    expect(loaded.draft).toBe("a = 1");
    expect(loaded.readAtMs).toBe(100);
    expect(isDirty(loaded)).toBe(false);
  });

  it("follows the disk while the draft is clean", () => {
    const next = receive(loaded, server("a = 3"), 200);
    expect(next.draft).toBe("a = 3");
    expect(next.diskChange).toBeNull();
  });

  it("holds a change on disk rather than replacing edits", () => {
    const next = receive(editing, server("a = 3"), 200);
    expect(next.draft).toBe("a = 2");
    expect(next.base?.content).toBe("a = 1");
    expect(next.diskChange).toMatchObject({
      hash: "h:a = 3",
      cause: "reload",
    });
    expect(next.readAtMs).toBe(200);
  });

  it("is idempotent, so it can run on every render", () => {
    const once = receive(editing, server("a = 3"), 200);
    expect(receive(once, server("a = 3"), 200)).toEqual(once);
  });

  it("only refreshes the read time for the same version", () => {
    const next = receive(editing, server("a = 1"), 300);
    expect(next.draft).toBe("a = 2");
    expect(next.diskChange).toBeNull();
    expect(next.readAtMs).toBe(300);
  });

  it("takes a disk version that already matches the draft", () => {
    const next = receive(editing, server("a = 2"), 200);
    expect(isDirty(next)).toBe(false);
    expect(next.base?.hash).toBe("h:a = 2");
  });
});

describe("saving", () => {
  it("is clean and saved once the save lands and the refetch arrives", () => {
    const afterSave = saved(editing, "a = 2");
    expect(isDirty(afterSave)).toBe(false);
    expect(afterSave.saved).toBe(true);
    const refetched = receive(afterSave, server("a = 2"), 200);
    expect(refetched.base?.hash).toBe("h:a = 2");
    expect(refetched.saved).toBe(true);
    expect(edit(refetched, "a = 4").saved).toBe(false);
  });

  it("keeps typing done before the refetch arrives", () => {
    const typed = edit(saved(editing, "a = 2"), "a = 5");
    const refetched = receive(typed, server("a = 2"), 200);
    expect(refetched.draft).toBe("a = 5");
    expect(refetched.diskChange).toBeNull();
    expect(refetched.base?.hash).toBe("h:a = 2");
  });

  it("fills in a rejected save's disk version when it is fetched", () => {
    const rejected = conflict(editing, "h:a = 3");
    expect(rejected.diskChange).toEqual({
      hash: "h:a = 3",
      server: null,
      cause: "conflict",
    });
    const fetched = receive(rejected, server("a = 3"), 200);
    expect(fetched.diskChange?.cause).toBe("conflict");
    expect(fetched.diskChange?.server?.content).toBe("a = 3");
  });
});

describe("resolving a disk change", () => {
  const changed: DraftState = receive(editing, server("a = 3"), 200);

  it("discards the draft for the version on disk", () => {
    const next = discard(changed);
    expect(next.draft).toBe("a = 3");
    expect(next.base?.hash).toBe("h:a = 3");
    expect(next.diskChange).toBeNull();
  });

  it("keeps the draft, building on the version on disk", () => {
    const next = keepDraft(changed);
    expect(next.draft).toBe("a = 2");
    expect(next.base?.hash).toBe("h:a = 3");
    expect(isDirty(next)).toBe(true);
    expect(next.diskChange).toBeNull();
  });

  it("keeps the draft over a version known only by its hash", () => {
    const next = keepDraft(conflict(editing, "h:new"));
    expect(next.base?.hash).toBe("h:new");
    expect(next.base?.content).toBe("a = 1");
    expect(next.draft).toBe("a = 2");
  });

  it("discards back to the base when the disk version is not here yet", () => {
    const next = discard(conflict(editing, "h:new"));
    expect(next.draft).toBe("a = 1");
  });
});
