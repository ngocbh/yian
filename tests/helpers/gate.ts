/**
 * The one way tests give a run a `beforeToolCall` hook, so they keep working when the hook
 * becomes a `runFlow` and `advance` option.
 */

import { type BeforeToolCall, GATE } from "../../src/tools/gate.ts";

/** Return a copy of `options` with `hook` as its `beforeToolCall`. */
export function gated<T extends object>(options: T, hook: BeforeToolCall | null): T {
  return { ...options, [GATE]: hook };
}
