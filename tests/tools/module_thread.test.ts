/**
 * A tool module's thread: started on the CODE worker entry, it imports its module once, lists its
 * tools with the caller's `current()`, keeps its port referenced while it waits, and rejects a
 * listing, naming the file, when the import throws or the thread exits. An idle thread never keeps
 * this process alive.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";

import { withEnv } from "../../src/defaults.ts";
import { runtime } from "../../src/platform.ts";
import { startToolModule, ToolModuleThread } from "../../src/tools/module_thread.ts";
import { listToolModule } from "../../src/tools/module_worker.ts";

const SOURCE = join(import.meta.dirname, "..", "..", "src");
const OK = 'export const ok = { name: "ok", description: "d", parameters: { type: "object" }, execute: () => 1 };\n';

let tmp: string;
const threads: ToolModuleThread[] = [];

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-module-thread-")));
});

afterEach(async () => {
  await Promise.all(threads.splice(0).map((thread) => thread.terminate()));
  rmSync(tmp, { recursive: true, force: true });
});

/** Write `source` as `crm.ts` in `tmp` and start its thread. */
function started(source: string): ToolModuleThread {
  const path = join(tmp, "crm.ts");
  writeFileSync(path, source);
  const thread = new ToolModuleThread(path);
  threads.push(thread);
  return thread;
}

test("a module importing yian's types only, from a folder without yian, lists as it does on this thread", async () => {
  const functionTools = pathToFileURL(join(SOURCE, "tools", "function_tools.ts")).href;
  const thread = started(`import type { ToolOptions } from "@ngocbh/yian/tools";
import { tool } from ${JSON.stringify(functionTools)};
export const lookup = {
  name: "lookup",
  description: "Find a customer by id.",
  parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
  isConcurrencySafe: () => true,
  execute: async (args) => ({ id: String(args.id) }),
} satisfies ToolOptions;
export const made = tool({ name: "made", description: "d", parameters: { type: "object" }, output: { type: "object" }, execute: () => ({}) });
export const find = { name: "lookup", description: "d", parameters: { type: "object" }, execute: () => 1 };
export const LIMIT = 10;
`);

  const listing = await thread.list();

  expect(thread.label).toBe("crm.ts");
  expect(listing).toEqual(listToolModule(await import(pathToFileURL(thread.path).href), "crm.ts").listing);
  expect(listing.tools.map((tool) => [tool.name, tool.hasIsConcurrencySafe, tool.outputJsonSchema])).toEqual([
    ["lookup", true, null],
    ["made", false, { type: "object" }],
  ]);
  expect(listing.refused).toEqual([
    { name: "find", message: "crm.ts export 'find' holds a tool named 'lookup'; rename one" },
  ]);
  expect(listing.others).toEqual(["LIMIT"]);
});

test("an import that throws rejects every listing, naming the file", async () => {
  const thread = started('throw new Error("boom");\n');

  await expect(thread.list()).rejects.toThrow("crm.ts failed to import: Error: boom");
  await expect(thread.list()).rejects.toThrow("crm.ts failed to import: Error: boom");
  expect(thread.exitCode).toBeNull();
});

test("a module that exits its thread rejects the listing with the exit code, and every later one", async () => {
  const thread = started("process.exit(3);\n");

  await expect(thread.list()).rejects.toThrow("crm.ts's thread exited (code 3)");
  expect(thread.exitCode).toBe(3);
  await expect(thread.list()).rejects.toThrow("crm.ts's thread exited (code 3)");
});

test("an import awaiting a promise nothing resolves keeps its thread alive until it is terminated", async () => {
  const thread = started("await new Promise(() => {});\nexport const LIMIT = 1;\n");

  const listing = thread.list();
  const first = await Promise.race([listing.then(() => "settled", () => "settled"), sleep(300, "waiting")]);

  expect(first).toBe("waiting");
  expect(thread.exitCode).toBeNull();
  await thread.terminate();
  await expect(listing).rejects.toThrow("crm.ts's thread exited (code ");
});

test("the thread imports its module once and lists each request with the caller's withEnv values", async () => {
  const log = join(tmp, "imports.log");
  const deep = '{ type: "object", properties: { a: { type: "object", properties: { b: { type: "string" } } } } }';
  const thread = started(`import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(log)}, "imported\\n");
export const deep = { name: "deep", description: "d", parameters: ${deep}, execute: () => 1 };
`);
  const shallow = (name: string) => (name === "YIAN_MAX_VALUE_NESTING" ? "3" : undefined);

  const refused = await withEnv(shallow, () => thread.list());
  await sleep(100);
  const listed = await thread.list();

  expect(refused.refused.map((each) => each.message)).toEqual([
    expect.stringMatching(/^crm\.ts export 'deep' .*nests deeper than 3 levels$/),
  ]);
  expect(listed.tools.map((tool) => tool.name)).toEqual(["deep"]);
  expect([thread.exitCode, readFileSync(log, "utf8")]).toEqual([null, "imported\n"]);
});

test(
  "a thread left idle doesn't keep this process alive",
  () => {
    writeFileSync(join(tmp, "crm.ts"), OK);
    const script = join(tmp, "idle.mjs");
    const module = pathToFileURL(join(SOURCE, "tools", "module_thread.ts")).href;
    writeFileSync(
      script,
      `const { ToolModuleThread } = await import(${JSON.stringify(module)});
const listing = await new ToolModuleThread(${JSON.stringify(join(tmp, "crm.ts"))}).list();
console.log(listing.tools.map((tool) => tool.name).join());
`,
    );

    const result = spawnSync(process.execPath, [...(runtime === "bun" ? ["--no-install"] : []), script], {
      encoding: "utf8",
      env: { PATH: process.env.PATH ?? "", HOME: tmp },
      timeout: 50_000,
      killSignal: "SIGKILL",
    });

    expect(result.status, `signal: ${result.signal}\nstderr: ${result.stderr}`).toBe(0);
    expect(result.stdout.trim()).toBe("ok");
  },
  60_000,
);

/** A module that appends to `beat` every 20 ms once its import starts, and never finishes importing. */
function beating(beat: string): string {
  return `import { appendFileSync } from "node:fs";
setInterval(() => appendFileSync(${JSON.stringify(beat)}, "."), 20);
await new Promise(() => {});
`;
}

/** Whether `beat` was written and then stopped growing: its size 100 ms on and 200 ms later agree. */
async function stopped(beat: string): Promise<boolean> {
  await sleep(100);
  const before = statSync(beat).size;
  await sleep(200);
  return before > 0 && statSync(beat).size === before;
}

test("an import that throws, or outlasts YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS, is refused naming the file, and its thread ends", async () => {
  const beat = join(tmp, "beat");
  writeFileSync(join(tmp, "slow.ts"), beating(beat));
  writeFileSync(join(tmp, "bad.ts"), 'throw new Error("boom");\n');
  const short = (name: string) => (name === "YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS" ? "1" : undefined);
  const start = (file: string) => withEnv(short, () => startToolModule(join(tmp, file), file, new AbortController().signal));

  const failed = await start("bad.ts").catch((error) => error);
  expect([failed, failed.message]).toEqual([expect.any(RangeError), "bad.ts failed to import: Error: boom"]);
  await expect(start("slow.ts")).rejects.toThrow(
    new RangeError("slow.ts didn't import within 1 s (YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS)"),
  );
  expect(await stopped(beat)).toBe(true);
});
