/** Give a test run's options a `beforeToolCall` hook. */

import type { BeforeToolCall } from "../../src/tools/gate.ts";

/** Return a copy of `options` with `hook` as its `beforeToolCall` option; `null` drops one. */
export function gated<T extends object>(options: T, hook: BeforeToolCall | null): T & { beforeToolCall: BeforeToolCall | null } {
  return { ...options, beforeToolCall: hook };
}
