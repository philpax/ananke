// When Enter in the composer sends rather than starting a new line.

export type EnterPress = {
  readonly key: string;
  readonly shiftKey: boolean;
  /// An IME is composing; its Enter commits the composition.
  readonly isComposing: boolean;
  /// A phone or tablet, whose on-screen Enter is the only way to type a
  /// newline, so there it always does.
  readonly coarsePointer: boolean;
};

export function sendsOnEnter(press: EnterPress): boolean {
  return (
    press.key === "Enter" &&
    !press.shiftKey &&
    !press.isComposing &&
    !press.coarsePointer
  );
}
