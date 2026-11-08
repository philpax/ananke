import { act, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  installLiveFakes,
  renderLive,
  service,
} from "../../../test/renderLive.tsx";
import { DetailHeader } from "./DetailHeader.tsx";

beforeEach(installLiveFakes);
afterEach(() => vi.unstubAllGlobals());

function renderHeader(svc = service("qwen")) {
  return renderLive(
    <DetailHeader
      service={svc}
      titleId="title"
      listHidden={false}
      serviceCount={1}
      onBrowse={() => {}}
    />,
  );
}

describe("DetailHeader", () => {
  it("says why a disabled service is disabled and offers Enable", () => {
    renderHeader(
      service("qwen", "disabled", { disabled_reason: "health_timeout" }),
    );
    expect(screen.getByText("health timeout")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enable" })).toBeEnabled();
    expect(
      screen.queryByRole("button", { name: "Start" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Chat/ })).toBeDisabled();
  });

  it("shows a start the daemon could not perform as an error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              status: "unavailable",
              error: {
                code: "insufficient_capacity",
                message: "no device has room",
                type: "server_error",
              },
            }),
            { status: 202 },
          ),
        ),
      ),
    );
    renderHeader();
    await act(async () => {
      screen.getByRole("button", { name: "Start" }).click();
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "no device has room",
    );
  });

  it("does not offer chat for a service outside the OpenAI API", () => {
    renderHeader(service("comfy", "idle", { openai_compat: false }));
    expect(screen.getByRole("button", { name: /^Chat/ })).toBeDisabled();
    expect(
      screen.getByText("Not on the OpenAI API, so it cannot be chatted with."),
    ).toBeInTheDocument();
  });
});
