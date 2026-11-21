/**
 * The one way tests give a run a `beforeToolCall` hook, so they keep working when the hook
 * becomes a `runFlow` and `advance` option.
 */

import { type BeforeToolCall, withGate } from "../../src/tools/gate.ts";

/** Return `options` with `hook` as its `beforeToolCall`; pass the returned object to the call. */
export function gated<T extends object>(options: T, hook: BeforeToolCall | null): T {
  return withGate(options, hook);
}
