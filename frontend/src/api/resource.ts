import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";

import { connectionStatus, watch } from "./socket";
import type { Source } from "./socket";

export type Resource<T> = {
  readonly data?: T;
  readonly loading: boolean;
  readonly error?: string;
};

export function useResource<T>(key: string, source: Source<T>): Resource<T> {
  const sourceRef = useRef(source);
  useLayoutEffect(() => {
    sourceRef.current = source;
  }, [source]);
  const [state, setState] = useState<Resource<T>>({ loading: true });
  useEffect(() => {
    setState({ loading: true });
    const stop = watch(sourceRef.current, (result) => {
      setState((previous) =>
        result.kind === "ok"
          ? { loading: false, data: result.value }
          : { ...previous, loading: false, error: result.message },
      );
    });
    const disconnect = connectionStatus((connected) => {
      if (!connected)
        setState((previous) => ({
          ...previous,
          error: "Dashboard: reconnecting.",
        }));
    });
    return () => {
      stop();
      disconnect();
    };
  }, [key]);
  return state;
}
