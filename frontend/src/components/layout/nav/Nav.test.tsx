import { act, screen, within } from "@testing-library/react";
import { Link } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  installLiveFakes,
  renderLive,
  service,
  setServices,
} from "../../../test/renderLive.tsx";
import { Nav } from "./Nav.tsx";

beforeEach(installLiveFakes);
afterEach(() => vi.unstubAllGlobals());

function sections() {
  return within(screen.getByRole("navigation", { name: "Sections" }));
}

describe("Nav", () => {
  it("links every section at every width", () => {
    renderLive(<Nav />);
    for (const name of ["Services", "Chat", "Events", "Stats", "Config"]) {
      expect(sections().getByRole("link", { name })).toBeInTheDocument();
    }
  });

  it("marks Services current on a service's page", () => {
    renderLive(<Nav />, { route: "/services/qwen" });
    expect(sections().getByRole("link", { name: "Services" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(sections().getByRole("link", { name: "Chat" })).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("counts services that need attention on the Services item", () => {
    renderLive(<Nav />, {
      services: [
        service("a", "failed"),
        service("b", "running"),
        service("c", "disabled", { disabled_reason: "oom" }),
        service("d", "disabled", { disabled_reason: "user_disabled" }),
      ],
    });
    expect(
      sections().getByRole("link", {
        name: "Services 2 services need attention",
      }),
    ).toBeInTheDocument();
  });

  it("carries the last chat service into Chat after leaving it", () => {
    renderLive(
      <>
        <Nav />
        <Link to="/events">away</Link>
      </>,
      { route: "/services/qwen", services: [service("qwen")] },
    );
    act(() => screen.getByRole("link", { name: "away" }).click());
    expect(sections().getByRole("link", { name: "Chat" })).toHaveAttribute(
      "href",
      "/chat?model=qwen",
    );
  });

  it("forgets the last service once it leaves the list", async () => {
    const { client } = renderLive(<Nav />, {
      route: "/services/qwen",
      services: [service("qwen")],
    });
    await setServices(client, [service("other")]);
    expect(sections().getByRole("link", { name: "Chat" })).toHaveAttribute(
      "href",
      "/chat",
    );
  });
});
