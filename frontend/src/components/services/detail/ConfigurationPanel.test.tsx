// Regenerating types makes a new backend field typed; it does not make it
// visible. These check that a container's identity and a command service's
// settings actually reach the panel.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { serviceDetail } from "../../../test/fixtures.ts";
import { ConfigurationPanel } from "./ConfigurationPanel.tsx";

describe("ConfigurationPanel", () => {
  it("shows runtime, image, and network for a container service", () => {
    render(
      <ConfigurationPanel
        detail={serviceDetail({
          container: {
            runtime: "podman",
            image: "ghcr.io/ggml-org/llama.cpp:server-cuda",
            network: "host",
          },
        })}
      />,
    );
    expect(
      screen.getByText("podman · ghcr.io/ggml-org/llama.cpp:server-cuda"),
    ).toBeInTheDocument();
    expect(screen.getByText("host")).toBeInTheDocument();
  });

  it("shows the live identity only while a container is running", () => {
    const { rerender } = render(
      <ConfigurationPanel
        detail={serviceDetail({
          container: {
            runtime: "docker",
            image: "ninfer:local",
            network: "host",
            container_id: "abcdef0123456789abcdef",
            container_name: "ananke-ninfer-42",
          },
        })}
      />,
    );
    // The ID is abbreviated the way both runtimes display it.
    expect(screen.getByText("abcdef012345")).toBeInTheDocument();
    expect(screen.getByText("ananke-ninfer-42")).toBeInTheDocument();

    rerender(
      <ConfigurationPanel
        detail={serviceDetail({
          container: {
            runtime: "docker",
            image: "ninfer:local",
            network: "bridge",
          },
        })}
      />,
    );
    expect(screen.queryByText("Container ID")).not.toBeInTheDocument();
  });

  it("leaves a native service without container rows", () => {
    render(<ConfigurationPanel detail={serviceDetail()} />);
    expect(screen.getByText("llama-cpp")).toBeInTheDocument();
    expect(screen.getByText(":48202")).toBeInTheDocument();
    expect(screen.queryByText("Container runtime")).not.toBeInTheDocument();
  });

  it("shows a command service's settings", () => {
    render(
      <ConfigurationPanel
        detail={serviceDetail({
          template: "command",
          openai_compat: false,
          command: {
            upstream_model: "flux",
            shutdown_command: ["comfy", "stop"],
            private_port_fixed: true,
            workdir: "/srv/comfy",
          },
        })}
      />,
    );
    expect(screen.getByText("not listed")).toBeInTheDocument();
    expect(screen.getByText("flux")).toBeInTheDocument();
    expect(screen.getByText(":48202 (fixed)")).toBeInTheDocument();
    expect(screen.getByText("/srv/comfy")).toBeInTheDocument();
    expect(screen.getByText("comfy stop")).toBeInTheDocument();
  });
});
