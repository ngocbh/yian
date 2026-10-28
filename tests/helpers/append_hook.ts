/**
 * Scoped hooks on `JsonlRunLog.prototype.append`, for tests that simulate a crash by refusing an
 * append.
 *
 * `append` is wrapped once, when this module loads; the wrapper runs the hooks of the
 * `withAppendHook` calls whose async work it runs in, then the engine's own append. Outside every
 * call it is the engine's append, so a test still running after it timed out can't refuse another
 * test's appends, and there is no patch to put back.
 *
 * Runs under Bun from source and under Node from `dist/`; only `node:` APIs.
 */

import { AsyncLocalStorage } from "node:async_hooks";

import { JsonlRunLog, type RecordEnvelope, type RunRecord } from "../../src/runtime/log.ts";

/** Called before an append in scope; throw to refuse it. */
export type AppendHook = (log: JsonlRunLog, record: RunRecord) => void;

/** The hook in effect for the current async work, if any. */
const scope = new AsyncLocalStorage<AppendHook>();

const engineAppend = JsonlRunLog.prototype.append;
JsonlRunLog.prototype.append = function (this: JsonlRunLog, record: RunRecord): RecordEnvelope {
  scope.getStore()?.(this, record);
  return engineAppend.call(this, record);
};

/**
 * Run `body` with `hook` called before every append it and the async work it starts make, after
 * the hook of any enclosing call; return what `body` returns.
 */
export function withAppendHook<T>(hook: AppendHook, body: () => T): T {
  const outer = scope.getStore();
  const hooks: AppendHook =
    outer === undefined
      ? hook
      : (log, record) => {
          outer(log, record);
          hook(log, record);
        };
  return scope.run(hooks, body);
}
