import { useEffect, useState } from "preact/hooks";

export type Section = "services" | "chat" | "events" | "stats" | "config";
export type Route = {
  readonly section: Section;
  readonly service?: string;
  readonly model?: string;
};

export function useNavigation() {
  const [route, setRoute] = useState(readRoute);
  useEffect(() => {
    const change = () => setRoute(readRoute());
    window.addEventListener("popstate", change);
    return () => window.removeEventListener("popstate", change);
  }, []);
  function navigate(next: Route) {
    const path =
      next.section === "services"
        ? next.service
          ? `/services/${encodeURIComponent(next.service)}`
          : "/"
        : `/${next.section}${next.model ? `?model=${encodeURIComponent(next.model)}` : ""}`;
    history.pushState(null, "", path);
    setRoute(next);
  }
  return { route, navigate };
}

function readRoute(): Route {
  if (location.pathname.startsWith("/services/")) {
    try {
      return {
        section: "services",
        service: decodeURIComponent(location.pathname.slice(10)),
      };
    } catch {
      return { section: "services" };
    }
  }
  const section = location.pathname.slice(1);
  if (
    section === "chat" ||
    section === "events" ||
    section === "stats" ||
    section === "config"
  )
    return {
      section,
      model: new URLSearchParams(location.search).get("model") ?? undefined,
    };
  return { section: "services" };
}
