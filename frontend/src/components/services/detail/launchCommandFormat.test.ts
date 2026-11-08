// The launch preview is a copy-pasteable command, so its shell quoting and
// its command/comment structure are correctness concerns, not cosmetics: a
// preview a user pastes has to do what it claims and nothing more.

import { describe, expect, it } from "vitest";

import type { LaunchCommand } from "../../../api/client.ts";
import { renderCommand, shellQuote } from "./launchCommandFormat.ts";

const processCommand: LaunchCommand = {
  source: "preview",
  argv: ["llama-server", "-m", "/models/x.gguf", "--host", "127.0.0.1"],
  env: [{ key: "CUDA_VISIBLE_DEVICES", value: "0" }],
  env_inherit: true,
};

const containerCommand: LaunchCommand = {
  source: "preview",
  argv: ["ninfer-serve", "/artifacts/model.ninfer"],
  env: [],
  env_inherit: false,
  container: {
    runtime: "docker",
    image: "ninfer:local",
    name_pattern: "ananke-ninfer-<run-id>",
    argv: ["ninfer-serve", "/artifacts/model.ninfer"],
    env: [],
    env_passthrough: ["HF_TOKEN"],
    mounts: [{ source: "/host/ninfer", target: "/artifacts", read_only: true }],
    network: "host",
    publication: null,
    ipc: "private",
    gpu_devices: ["nvidia.com/gpu=0"],
    create_argv: [
      "docker",
      "create",
      "--name",
      "ananke-ninfer-0",
      "--network",
      "host",
      "-v",
      "/host/ninfer:/artifacts:ro",
      "ninfer:local",
      "ninfer-serve",
      "/artifacts/model.ninfer",
    ],
  },
};

/** Lines that would actually execute, with comments and continuations gone. */
function runnableLines(rendered: string): string[] {
  return rendered
    .split("\\\n")
    .join("")
    .split("\n")
    .filter((l) => !l.trimStart().startsWith("#"))
    .filter((l) => l.trim().length > 0);
}

describe("shellQuote", () => {
  it("leaves safe tokens alone", () => {
    expect(shellQuote("--model")).toBe("--model");
    expect(shellQuote("/models/x.gguf")).toBe("/models/x.gguf");
    expect(shellQuote("nvidia.com/gpu=0")).toBe("nvidia.com/gpu=0");
  });

  it("quotes whitespace and shell metacharacters", () => {
    expect(shellQuote("/my models/x.gguf")).toBe("'/my models/x.gguf'");
    expect(shellQuote('{"canvas_length": 256}')).toBe(
      "'{\"canvas_length\": 256}'",
    );
    expect(shellQuote("")).toBe("''");
  });

  it("escapes embedded single quotes", () => {
    expect(shellQuote("it's")).toBe("'it'\\''s'");
  });
});

describe("renderCommand for a process workload", () => {
  it("renders the binary and its arguments", () => {
    const out = renderCommand(processCommand);
    expect(out).toContain("llama-server");
    expect(out).toContain("-m /models/x.gguf");
    expect(out).toContain("CUDA_VISIBLE_DEVICES=0");
  });

  it("quotes a model path containing spaces", () => {
    const out = renderCommand({
      ...processCommand,
      argv: ["llama-server", "-m", "/my models/x.gguf"],
    });
    expect(out).toContain("'/my models/x.gguf'");
  });
});

describe("renderCommand for a container workload", () => {
  it("renders the runtime create command", () => {
    const out = renderCommand(containerCommand);
    expect(out).toContain("docker");
    expect(out).toContain("create");
    expect(out).toContain("--name ananke-ninfer-0");
    expect(out).toContain("/host/ninfer:/artifacts:ro");
    expect(out).toContain("ninfer:local");
  });

  it("shows the in-container command as a comment, not a second command", () => {
    // Pasting this must create the container and nothing else. Rendering
    // the in-container argv as its own line would run the workload on the
    // host.
    const out = renderCommand(containerCommand);
    expect(out).toContain("# in-container command: ninfer-serve");

    const runnable = runnableLines(out);
    expect(runnable).toHaveLength(1);
    expect(runnable[0]!.trimStart().startsWith("docker")).toBe(true);
  });

  it("never expands a passthrough secret", () => {
    const out = renderCommand({
      ...containerCommand,
      container: {
        ...containerCommand.container!,
        create_argv: [
          ...containerCommand.container!.create_argv,
          "-e",
          "HF_TOKEN",
        ],
      },
    });
    // The name is present; a value is never rendered because the preview
    // does not carry one.
    expect(out).toContain("HF_TOKEN");
    expect(out).not.toContain("HF_TOKEN=");
  });

  it("quotes a mount path containing spaces", () => {
    const out = renderCommand({
      ...containerCommand,
      container: {
        ...containerCommand.container!,
        create_argv: ["docker", "create", "-v", "/my models:/models:ro", "img"],
      },
    });
    expect(out).toContain("'/my models:/models:ro'");
  });
});
