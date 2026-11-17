/**
 * Scoped hooks on the calls a run sends to its tool modules' threads, for tests that count the
 * tool calls a run made: a module's tool runs on its own thread, where no stub tool sees it.
 *
 * `ToolModuleThread.prototype.call` is wrapped once, when this module loads; the wrapper runs the
 * hooks of the `withModuleCalls` calls whose async work it runs in, then sends the call. Outside
 * every call it is the thread's own `call`, so nothing outside a test's scope changes.
 *
 * Runs under Bun from source and under Node from `dist/`; only `node:` APIs.
 */

import { AsyncLocalStorage } from "node:async_hooks";

import type { ToolContext } from "../../src/tools/context.ts";
import { ToolModuleThread } from "../../src/tools/module_thread.ts";

/**
 * Called just before a module tool call is sent to its thread, with the call's context, the
 * module's key (as a run records its tools under `tool_modules`) and the tool's export name.
 */
export type ModuleCallHook = (ctx: ToolContext, module: string, tool: string) => void;

/** The hook in effect for the current async work, if any. */
const scope = new AsyncLocalStorage<ModuleCallHook>();

const ownCall = ToolModuleThread.prototype.call;
ToolModuleThread.prototype.call = function (this: ToolModuleThread, ...args: Parameters<typeof ownCall>) {
  const [tool, , ctx] = args;
  scope.getStore()?.(ctx, this.label, tool);
  return ownCall.apply(this, args);
};

/**
 * Run `body` with `hook` called for every module tool call it and the async work it starts send,
 * after the hook of any enclosing call; return what `body` returns.
 */
export function withModuleCalls<T>(hook: ModuleCallHook, body: () => T): T {
  const outer = scope.getStore();
  const hooks: ModuleCallHook =
    outer === undefined
      ? hook
      : (ctx, module, tool) => {
          outer(ctx, module, tool);
          hook(ctx, module, tool);
        };
  return scope.run(hooks, body);
}
