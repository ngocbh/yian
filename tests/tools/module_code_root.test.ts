/**
 * A tool module's thread under a code root: every file the module's imports load, by path or by
 * package name, must have its real path inside the root, and built-ins always load. Each route out
 * fails the module's import, naming the importer and the import, and so does a call that could undo
 * the guard; the imports that stay inside, and type-only ones, load; a module without a root loads
 * as before; a root whose real path changed is refused; and a fresh thread that takes over after an
 * exit is held to the root too.
 */

import { afterAll, afterEach, expect, test } from "bun:test";
import { rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { runtime } from "../../src/platform.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { startToolModule, type ToolModuleThread } from "../../src/tools/module_thread.ts";
import { ALLOWED_IMPORTS, codeRootTree, ESCAPE_ROUTES, refuses } from "../helpers/code_root_routes.ts";

const tree = codeRootTree();
afterAll(() => rmSync(tree.base, { recursive: true, force: true }));

const threads: ToolModuleThread[] = [];
afterEach(async () => {
  await Promise.all(threads.splice(0).map((thread) => thread.terminate()));
});

/** A tool whose description is the `got` its module bound. */
const SHOWS_GOT = 'export const ok = { name: "ok", description: String(got), parameters: { type: "object" }, execute: () => 1 };\n';

/**
 * Write `source` as `entry.ts` in the root and start its thread under `codeRoot`; return what the
 * listing describes its first tool as.
 */
async function described(source: string, codeRoot: string | null = tree.root): Promise<string> {
  const path = join(tree.root, "entry.ts");
  writeFileSync(path, source);
  const { thread, listing } = await startToolModule(path, "entry.ts", new AbortController().signal, codeRoot);
  threads.push(thread);
  return listing.tools[0]!.description;
}

/** What `promise` rejects with; it must reject. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
  return await promise.then(
    (value) => {
      throw new Error(`resolved with ${String(value)}`);
    },
    (error: Error) => error,
  );
}

for (const route of ESCAPE_ROUTES) {
  const refusal = route.refusal[runtime];
  test.skipIf(refusal === null)(`under a code root, a module fails its import through ${route.name}`, async () => {
    const error = await rejection(described(route.source(tree) + SHOWS_GOT));
    expect(error.message).toStartWith("entry.ts failed to import: ");
    expect(refuses(refusal!, error.message), error.message).toBe(true);
  });
}

for (const allowed of ALLOWED_IMPORTS) {
  const skip = allowed.bunOnly === true && runtime !== "bun";
  test.skipIf(skip)(`a module importing ${allowed.name} loads under the code root`, async () => {
    expect(await described(allowed.source(tree) + SHOWS_GOT)).toBe(allowed.got);
  });
}

test("an addon inside the code root goes on to the runtime's own loader", async () => {
  const error = await rejection(described('import { createRequire } from "node:module";\nconst got = typeof createRequire(import.meta.url)("./inside.node");\n' + SHOWS_GOT));
  expect(error.message).toContain("inside.node");
  expect(error.message).not.toContain("code root");
});

test("a module thread refuses a code root whose real path changed since the flow was loaded", async () => {
  const error = await rejection(described(SHOWS_GOT.replace("String(got)", '"d"'), join(tree.base, "rootlink")));
  expect(error.message).toContain("the code root's real path has changed since the flow was loaded");
});

test("without a code root, a module loads files from outside as before", async () => {
  expect(await described('import { x } from "../outside/evil.ts";\nconst got = x;\n' + SHOWS_GOT, null)).toBe("OUTSIDE");
});

test("a fresh thread that takes over after an exit is held to the code root too", async () => {
  const path = join(tree.root, "entry.ts");
  writeFileSync(
    path,
    `export const escape = { name: "escape", description: "d", parameters: { type: "object" }, execute: async () => (await import("../out" + "side/evil.ts")).x };
export const exit = { name: "exit", description: "d", parameters: { type: "object" }, execute: () => process.exit(2) };
`,
  );
  const { thread } = await startToolModule(path, "entry.ts", new AbortController().signal, tree.root);
  threads.push(thread);
  const call = async (tool: string) => {
    const checked = await thread.check(tool, tool, {});
    if (checked.issues !== undefined) throw new Error("issues");
    const fields = { callId: "r1/a/1/0", attempt: 1, toolName: tool, runId: "r1", node: "a", context: undefined };
    return await thread.call(tool, checked.value, new ToolContext({ ...fields, signal: new AbortController().signal }));
  };
  const refused = "entry.ts imports '../outside/evil.ts', which resolves outside the code root";

  expect((await rejection(call("escape"))).message).toContain(refused);
  await rejection(call("exit"));
  expect((await rejection(call("escape"))).message).toContain(refused);
});
