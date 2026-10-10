export function Message({
  error,
  loading,
  empty,
}: {
  readonly error?: string;
  readonly loading?: boolean;
  readonly empty?: string;
}) {
  if (error)
    return (
      <p role="alert" class="py-2 text-[13px] wrap-anywhere text-danger">
        {error}
      </p>
    );
  return (
    <p role="status" class="py-2 text-[13px] text-muted">
      {loading ? "Loading…" : empty}
    </p>
  );
}
