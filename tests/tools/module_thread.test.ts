/**
 * A tool module's thread: started on the CODE worker entry, it imports its module once, lists its
 * tools with the caller's `current()`, keeps its port referenced while it waits, and rejects a
 * listing, naming the file, when the import throws or the thread exits. An idle thread never keeps
 * this process alive. A check runs the tool's validator and `isConcurrencySafe` there and keeps the
 * validator's output for the call, until the call settles or its handle here is collected. A call
 * runs its tool there with the call's context; its throw comes back with its tool error kinds, and a
 * result that isn't a yian value as a failed check.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";
import v8 from "node:v8";
import vm from "node:vm";

import { withEnv } from "../../src/defaults.ts";
import { runtime } from "../../src/platform.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { isToolCheckError, isTransientToolError, toolErrorKinds } from "../../src/tools/errors.ts";
import { type CheckedArgs, startToolModule, ToolModuleThread } from "../../src/tools/module_thread.ts";
import { listToolModule } from "../../src/tools/module_worker.ts";
import { FileRef } from "../../src/typesys/values.ts";

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

/** A call's context from run `r1`'s node `a`, stopped by `signal`. */
function context(signal = new AbortController().signal, attempt = 2): ToolContext {
  return new ToolContext({ callId: "r1/a/1/0", attempt, toolName: "find", runId: "r1", node: "a", context: { app: 1 }, signal });
}

/** Check `args` for `tool` on `thread`, as the registry does, and return the checked arguments. */
async function checked(thread: ToolModuleThread, tool: string, args: Record<string, unknown>): Promise<CheckedArgs> {
  const result = await thread.check(tool, tool, args);
  if (result.issues !== undefined) throw new Error(`issues: ${JSON.stringify(result.issues)}`);
  return result.value;
}

/** Check `args` for `tool` on `thread` and call it with them, as the registry does. */
async function called(thread: ToolModuleThread, tool: string, args: Record<string, unknown>, ctx = context()): Promise<unknown> {
  return await thread.call(tool, await checked(thread, tool, args), ctx);
}

/** The source of a plain-object tool `name` whose `execute` is `body`. */
function calling(name: string, body: string): string {
  return `export const ${name} = { name: "${name}", description: "d", parameters: { type: "object" }, ${body} };\n`;
}

test("a call runs its tool on the thread, bound to its object, with the call's context and files crossing both ways", async () => {
  const thread = started(
    calling(
      "echo",
      `execute(args, ctx) {
    const { callId, attempt, toolName, runId, node, context, signal } = ctx;
    return { self: this === echo, ctx: { callId, attempt, toolName, runId, node, context: context ?? null, aborted: signal.aborted }, args };
  }`,
    ) + calling("nothing", "execute: () => undefined"),
  );
  const file = new FileRef({ uri: "file:///w/a.txt", mime: "text/plain", name: "a.txt" });
  const args = { file, big: 2n ** 60n, list: [file, 1.5] };

  const echoed = await called(thread, "echo", args);

  expect(echoed).toEqual({
    self: true,
    ctx: { callId: "r1/a/1/0", attempt: 2, toolName: "echo", runId: "r1", node: "a", context: null, aborted: false },
    args,
  });
  expect(FileRef.isFileRef((echoed as { args: { file: unknown } }).args.file)).toBe(true);
  expect(await called(thread, "nothing", {})).toBeNull();
});

test("a tool's throw comes back with its name, message, stack and tool error kinds; any other value as an Error", async () => {
  const thread = started(`const TOOL_ERROR = Symbol.for("yian.tool_error");
${calling(
  "fail",
  `execute({ how }) {
    const marked = (kinds, extra = {}) => Object.assign(new Error(how), { [TOOL_ERROR]: kinds }, extra);
    if (how === "reported") throw marked(["reported"]);
    if (how === "reported permanent") throw marked(["reported", "permanent"]);
    if (how === "permanent") throw marked(["permanent"]);
    if (how === "check") throw marked(["check", "unknown"]);
    if (how === "transient") throw marked(["transient"], { retryAfter: 5 });
    if (how === "type") throw new TypeError("bad id");
    if (how === "text") throw "boom";
    throw { code: 7 };
  }`,
)}`);
  const thrown = (how: string) => called(thread, "fail", { how }).then(() => null, (error: Error) => error);

  for (const kinds of [["reported"], ["reported", "permanent"], ["permanent"], ["check"]] as const) {
    const error = await thrown(kinds.join(" "));
    expect([error?.message, toolErrorKinds(error)]).toEqual([kinds.join(" "), kinds]);
  }
  const transient = (await thrown("transient")) as Error & { retryAfter?: unknown };
  expect([isTransientToolError(transient), transient.retryAfter]).toEqual([true, 5]);
  const typed = await thrown("type");
  expect([typed instanceof Error, typed?.name, typed?.message, toolErrorKinds(typed)]).toEqual([true, "TypeError", "bad id", []]);
  expect(typed?.stack).toContain("crm.ts");
  const text = await thrown("text");
  expect([text instanceof Error, text?.message, text?.cause]).toEqual([true, "boom", "boom"]);
  const record = await thrown("record");
  expect([record?.message, record?.cause]).toEqual(["[object Object]", { code: 7 }]);
});

test("a result that isn't a yian value, or that pends, fails the call's check, naming the export", async () => {
  const thread = started(
    calling(
      "bad",
      `execute({ how }) {
    if (how === "date") return new Date(0);
    if (how === "map") return { a: [new Map()] };
    if (how === "deep") { let value = 1; for (let i = 0; i < 1500; i++) value = [value]; return value; }
    return { [Symbol.for("yian.pending")]: true };
  }`,
    ),
  );
  const refused = (how: string) => called(thread, "bad", { how }).then(() => null, (error: Error) => error);

  const date = await refused("date");
  expect([isToolCheckError(date), date?.name, date?.message]).toEqual([true, "ToolCheckError", "crm.ts export 'bad' result: Date is not a yian value"]);
  const map = await refused("map");
  expect([map?.message, (map?.cause as { path?: unknown }).path]).toEqual(["crm.ts export 'bad' result at a[0]: Map is not a yian value", ["a", 0]]);
  expect((await refused("deep"))?.message).toMatch(/^crm\.ts export 'bad' result at (\[0\])+: value nests deeper than 100 levels/);
  const pends = await refused("pending");
  expect([isToolCheckError(pends), pends?.message]).toEqual([true, "crm.ts export 'bad' returned pending(), which a tool module can't do yet"]);
});

test("a stop aborts the tool's own signal on the thread, and other calls go on meanwhile", async () => {
  const thread = started(
    calling("wait", 'execute: (_args, ctx) => new Promise((resolve) => ctx.signal.addEventListener("abort", () => resolve("stopped")))') +
      calling("quick", 'execute: () => "quick"'),
  );
  const stop = new AbortController();

  const waiting = called(thread, "wait", {}, context(stop.signal));
  expect(await called(thread, "quick", {})).toBe("quick");
  stop.abort();

  expect(await waiting).toBe("stopped");
});

/** Source of a full garbage collection on the running thread: `Bun.gc` on Bun; on Node, V8's `gc`, exposed for the call. */
const COLLECT = `import v8 from "node:v8";
import vm from "node:vm";
const collect = () => {
  if (typeof Bun !== "undefined") return Bun.gc(true);
  v8.setFlagsFromString("--expose-gc");
  vm.runInNewContext("gc")();
};
`;

/** A full garbage collection on this thread, as `COLLECT` runs one. */
function collectGarbage(): void {
  if (typeof Bun !== "undefined") {
    Bun.gc(true);
    return;
  }
  v8.setFlagsFromString("--expose-gc");
  (vm.runInNewContext("gc") as () => void)();
}

/**
 * A module whose `kept` validator numbers each output it makes, from 1; its call records each
 * attempt in that output and fails its first attempt with the tool error kind `fail`, if any.
 * `collected` lists the outputs the thread collected.
 */
const KEPT = `${COLLECT}const TOOL_ERROR = Symbol.for("yian.tool_error");
const parameters = { type: "object", properties: { fail: { type: "string" } } };
let made = 0;
const collected = [];
const outputs = new FinalizationRegistry((n) => collected.push(n));
const checker = {
  "~standard": {
    version: 1,
    vendor: "t",
    jsonSchema: { input: () => parameters },
    validate(value) {
      const output = { ...value, n: ++made, attempts: [] };
      outputs.register(output, made);
      return { value: output };
    },
  },
};
export const kept = {
  name: "kept",
  description: "d",
  parameters: checker,
  isConcurrencySafe: (args) => args.n === 1,
  execute(args, ctx) {
    args.attempts.push(ctx.attempt);
    if (args.fail !== undefined && ctx.attempt === 1) throw Object.assign(new Error(args.fail), { [TOOL_ERROR]: [args.fail] });
    return { n: args.n, attempts: args.attempts };
  },
};
${calling("gone", "async execute() { collect(); await new Promise((resolve) => setTimeout(resolve, 10)); return collected.sort(); }")}`;

test("a check runs the validator and isConcurrencySafe on the thread, and each attempt's call gets its one output until one settles", async () => {
  const thread = started(KEPT);

  const first = await checked(thread, "kept", { fail: "transient" });
  const second = await checked(thread, "kept", {});
  expect([first.safe, second.safe]).toEqual([true, false]);

  // Not `rejects`: Bun's waits for a pending call in a way that stalls the next one.
  const message = (call: Promise<unknown>) => call.then(() => null, (error: Error) => error.message);
  expect(await message(thread.call("kept", first, context(undefined, 1)))).toBe("transient");
  expect(await thread.call("kept", first, context(undefined, 2))).toEqual({ n: 1, attempts: [1, 2] });
  expect(await message(thread.call("kept", first, context(undefined, 3)))).toBe(
    "crm.ts export 'kept': its checked arguments aren't kept on this thread",
  );
});

test("a call's output is released once it settles, unless a TOOL node may retry it, and once its handle is collected", async () => {
  const thread = started(KEPT);
  /** Check and call `kept`, failing with `fail`, and drop the handle. */
  const dropped = async (fail: string) => {
    await thread.call("kept", await checked(thread, "kept", { fail }), context(undefined, 1)).catch(() => null);
  };
  /** The outputs the thread collected, once it collected `want` (or after 5 s). */
  const gone = async (want: number[]) => {
    let seen: unknown = null;
    for (let tries = 0; tries < 100; tries++) {
      collectGarbage();
      await sleep(20);
      seen = await called(thread, "gone", {});
      if (JSON.stringify(seen) === JSON.stringify(want)) break;
    }
    return seen;
  };

  const settled = await checked(thread, "kept", {});
  await thread.call("kept", settled, context());
  const permanent = await checked(thread, "kept", { fail: "permanent" });
  await thread.call("kept", permanent, context(undefined, 1)).catch(() => null);
  const transient = await checked(thread, "kept", { fail: "transient" });
  await thread.call("kept", transient, context(undefined, 1)).catch(() => null);
  await dropped("transient");
  const stop = new AbortController();
  stop.abort();
  const stopped = await checked(thread, "kept", { fail: "transient" });
  await thread.call("kept", stopped, context(stop.signal, 1)).catch(() => null);

  expect(await gone([1, 2, 4, 5])).toEqual([1, 2, 4, 5]);
  // The handle still held keeps its output on the thread for the next attempt.
  expect(await thread.call("kept", transient, context())).toEqual({ n: 3, attempts: [1, 2] });
  expect([settled.safe, permanent.safe, stopped.safe]).toEqual([true, false, false]);
});
