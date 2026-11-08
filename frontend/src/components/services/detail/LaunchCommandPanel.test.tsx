// Process and container previews are visually distinct states of one panel.
// These assert what a user actually sees, so a discriminated variant can't
// silently fall through to the other's rendering.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { LaunchCommand } from "../../../api/client.ts";
import { CommandPanel } from "./LaunchCommandPanel.tsx";

const processCommand: LaunchCommand = {
  source: "preview",
  argv: ["llama-server", "-m", "/models/x.gguf"],
  env: [{ key: "CUDA_VISIBLE_DEVICES", value: "0" }],
  env_inherit: true,
};

const containerCommand: LaunchCommand = {
  source: "running",
  argv: ["ninfer-serve"],
  env: [],
  env_inherit: false,
  container: {
    runtime: "podman",
    image: "ninfer:local",
    name_pattern: "ananke-ninfer-<run-id>",
    argv: ["ninfer-serve"],
    env: [],
    env_passthrough: ["HF_TOKEN"],
    mounts: [],
    network: "host",
    publication: null,
    ipc: "private",
    gpu_devices: [],
    create_argv: ["podman", "create", "ninfer:local", "ninfer-serve"],
  },
};

describe("CommandPanel", () => {
  it("launch_command_section_renders_process_preview", () => {
    render(<CommandPanel label="Standalone" command={processCommand} />);

    // A native process advertises its environment-inheritance state, which
    // is the thing that decides what the child can see.
    expect(screen.getByText("env: inherited")).toBeInTheDocument();
    expect(screen.getByText(/llama-server/)).toBeInTheDocument();
    expect(screen.queryByText(/podman/)).not.toBeInTheDocument();
  });

  it("launch_command_section_renders_container_preview", () => {
    render(<CommandPanel label="Standalone" command={containerCommand} />);

    // A container advertises its runtime and image instead: env inheritance
    // does not apply to it, and showing that badge would be a lie.
    expect(screen.getByText("podman · ninfer:local")).toBeInTheDocument();
    expect(screen.queryByText("env: inherited")).not.toBeInTheDocument();
    expect(screen.queryByText("env: isolated")).not.toBeInTheDocument();
    expect(screen.getByText(/podman create/)).toBeInTheDocument();
  });

  it("container_preview_redacts_passthrough_secrets", () => {
    render(<CommandPanel label="Standalone" command={containerCommand} />);
    // The rendered panel carries no value for a passthrough variable,
    // because the wire payload never carried one.
    expect(document.body.textContent).not.toMatch(/HF_TOKEN=/);
  });

  it("renders nothing runnable when no command is available", () => {
    render(<CommandPanel label="Current conditions" command={null} />);
    expect(screen.getByText("Current conditions")).toBeInTheDocument();
  });
});
