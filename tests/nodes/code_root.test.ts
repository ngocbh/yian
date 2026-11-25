/**
 * CODE steps under a code root (`caps.codeRoot`): every file a `./file.ts:fn` reference's module or
 * an inline body imports, by path or by package name, must have its real path inside the root, and
 * built-ins always load. Each route out fails the step, naming the importer and the import, and so
 * does a call that could undo the guard, which stays for the thread's later calls; the imports that
 * stay inside, and type-only ones, load; a step without a root loads as before; a root whose real
 * path changed is refused; and a worker thread serves only the code root it started under.
 */

import { afterAll, expect, test } from "bun:test";
import { rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { Output } from "../../src/nodes/base.ts";
import { CodeError, CodeNode } from "../../src/nodes/code/node.ts";
import { runtime } from "../../src/platform.ts";
import { ALLOWED_IMPORTS, codeRootTree, ESCAPE_ROUTES, LOCKED, refuses } from "../helpers/code_root_routes.ts";

const tree = codeRootTree();
afterAll(() => rmSync(tree.base, { recursive: true, force: true }));

/** Run `code` as CODE node `n` in the root, under `codeRoot`, and return its value. */
async function ran(code: string, codeRoot: string | null = tree.root): Promise<unknown> {
  const node = new CodeNode("n", { code, baseDir: tree.root });
  const result = await node.run({}, codeRoot === null ? {} : { codeRoot });
  expect(result).toBeInstanceOf(Output);
  return (result as Output).value;
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

let entries = 0;

/**
 * Write `source` as a fresh module in the root, with a `run` export returning its `got`, and return
 * its file name: a thread imports a module once and keeps it, so each test's has a name of its own.
 */
function entry(source: string): string {
  const name = `entry-${++entries}.ts`;
  writeFileSync(join(tree.root, name), source + "export function run() { return got; }\n");
  return name;
}

/** `refusal`, which names the importer `entry.ts`, as it reads for the module `name`. */
function renamed(refusal: string | RegExp, name: string): string | RegExp {
  if (typeof refusal === "string") return refusal.replace("entry.ts", name);
  return new RegExp(refusal.source.replace("entry\\.ts", name.replace(".", "\\.")));
}

for (const route of ESCAPE_ROUTES) {
  const refusal = route.refusal[runtime];
  test.skipIf(refusal === null)(`under a code root, a reference fails when its module imports through ${route.name}`, async () => {
    const name = entry(route.source(tree));

    const error = await rejection(ran(`./${name}:run`));

    expect(error).toBeInstanceOf(CodeError);
    expect(refuses(renamed(refusal!, name), error.message), error.message).toBe(true);
  });
}

for (const allowed of ALLOWED_IMPORTS) {
  const skip = allowed.bunOnly === true && runtime !== "bun";
  test.skipIf(skip)(`a reference whose module imports ${allowed.name} runs under the code root`, async () => {
    expect(await ran(`./${entry(allowed.source(tree))}:run`)).toBe(allowed.got);
  });
}

test("an inline body's import() of a file outside the code root fails, naming the import", async () => {
  const error = await rejection(ran(`return (await import(${JSON.stringify(tree.evil)})).x;`));
  expect(error).toBeInstanceOf(CodeError);
  expect(error.message).toBe(`Error: the code imports '${tree.evil}', which resolves outside the code root`);
});

test("an inline body's import() of a data: URL fails", async () => {
  const error = await rejection(ran(`return (await import("data:text/javascript,export const x = 1")).x;`));
  expect(error.message).toBe("Error: the code imports a 'data:' URL, which isn't a file inside the code root");
});

test("an inline body imports a built-in and a file inside the code root", async () => {
  const inner = JSON.stringify(join(tree.root, "inner.ts"));
  expect(await ran(`const { sep } = await import("node:path");\nreturn sep + (await import(${inner})).x;`)).toBe("/INNER");
});

test("a body that tries to take the guard away leaves it for the thread's later calls", async () => {
  const undo = runtime === "bun" ? "Bun.plugin.clearAll()" : '(await import("node:module")).registerHooks({ resolve: (s, c, n) => n(s, c) })';
  const name = runtime === "bun" ? "Bun.plugin.clearAll" : "module.registerHooks";
  expect(await ran(`try { ${undo}; return "undone"; } catch (error) { return error.message; }`)).toBe(LOCKED(name));
  const error = await rejection(ran(`return (await import(${JSON.stringify(tree.evil)})).x;`));
  expect(error.message).toBe(`Error: the code imports '${tree.evil}', which resolves outside the code root`);
});

test("a CODE step refuses a code root whose real path changed since the flow was loaded", async () => {
  const error = await rejection(ran("return 1;", join(tree.base, "rootlink")));
  expect(error.message).toContain("the code root's real path has changed since the flow was loaded");
});

test("without a code root, a reference's module loads files from outside as before", async () => {
  const name = entry('import { x } from "../outside/evil.ts";\nconst got = x;\n');
  expect(await ran(`./${name}:run`, null)).toBe("OUTSIDE");
});

test("a CODE thread serves one code root: a call under another root, or none, takes another thread", async () => {
  const name = entry(
    'let calls = 0;\nexport async function count() {\n  calls += 1;\n' +
      '  const got = await import("../out" + "side/evil.ts").then((m) => m.x, (e) => e.message);\n' +
      "  return `${calls} ${got}`;\n}\nconst got = null;\n",
  );
  const refused = `${name} imports '../outside/evil.ts', which resolves outside the code root`;
  const count = `./${name}:count`;

  // The root's thread is reused, and keeps its module's count; the folder above it and no root
  // each take their own, where the file above the root loads.
  expect(await ran(count)).toBe(`1 ${refused}`);
  expect(await ran(count)).toBe(`2 ${refused}`);
  expect(await ran(count, tree.base)).toBe("1 OUTSIDE");
  expect(await ran(count, null)).toBe("1 OUTSIDE");
  expect(await ran(count)).toBe(`3 ${refused}`);
});
