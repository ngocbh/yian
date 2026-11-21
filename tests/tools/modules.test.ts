/**
 * A started tool module's checks and node scope: what a run refuses in a listing for the aliases
 * that reach it, and the stand-ins a node's scope holds for its tools, each keyed by its file and
 * export so two aliases of one file bind a tool once, and each checking and calling its export on
 * the module's thread, where its checks act as a host tool's do.
 */

import { expect, test } from "bun:test";
import { mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { pending, ToolContext } from "../../src/tools/context.ts";
import { isToolCheckError, toolErrorKinds } from "../../src/tools/errors.ts";
import { nodeScope, expandGrants, parseToolGrants } from "../../src/tools/grants.ts";
import { ToolModuleThread } from "../../src/tools/module_thread.ts";
import { listToolModule, type ListedTool, type ToolModuleListing } from "../../src/tools/module_worker.ts";
import {
  checkModuleListing,
  moduleKey,
  moduleScope,
  recordedModuleScope,
  type StartedToolModule,
} from "../../src/tools/modules.ts";
import { ToolSet } from "../../src/tools/registry.ts";

/** A listed tool `name` with an `id` parameter. */
function listed(name: string): ListedTool {
  const parametersJsonSchema = { type: "object", properties: { id: { type: "string" } } };
  return {
    name,
    description: `${name}.`,
    parametersJsonSchema,
    outputJsonSchema: null,
    resumable: name === "lookup",
    hasParametersValidator: false,
    hasOutputValidator: false,
    hasIsConcurrencySafe: false,
  };
}

const LISTING: ToolModuleListing = {
  tools: [listed("create"), listed("lookup")],
  refused: [{ name: "find", message: "crm.ts export 'find' holds a tool named 'lookup'; rename one" }],
  others: ["LIMIT", "default"],
};

test.each<[ToolModuleListing, { group: boolean; exports: string[] }, string | null]>([
  [LISTING, { group: false, exports: ["lookup", "create"] }, null],
  [LISTING, { group: false, exports: ["lookp"] }, "crm.ts has no export 'lookp'; did you mean 'lookup'?"],
  [LISTING, { group: false, exports: ["LIMIT"] }, "crm.ts export 'LIMIT' isn't a tool: export an object with name, description, parameters and execute"],
  [LISTING, { group: false, exports: ["find"] }, "crm.ts export 'find' holds a tool named 'lookup'; rename one"],
  [LISTING, { group: true, exports: [] }, "crm.ts export 'find' holds a tool named 'lookup'; rename one"],
  [{ ...LISTING, refused: [] }, { group: true, exports: [] }, null],
  [{ tools: [], refused: [], others: ["LIMIT"] }, { group: true, exports: [] }, "crm.ts exports no tools"],
  [{ ...LISTING, refused: [], tools: [listed("ask")] }, { group: true, exports: [] }, "crm.ts export 'ask' takes a name yian reserves; rename it"],
  [{ ...LISTING, refused: [], tools: [listed("ask")] }, { group: false, exports: ["ask"] }, null],
])("a listing checked for what reaches it: %#", (listing, reach, message) => {
  const check = () => checkModuleListing("crm.ts", listing, reach);
  if (message === null) expect(check).not.toThrow();
  else expect(check).toThrow(new RangeError(message));
});

test("a node scope of recorded tools holds what moduleScope would, and nothing the run didn't record", () => {
  const recorded = new Map([["/w/crm.ts", [{ name: "create" }, { name: "lookup" }]]]);
  const refs = [
    { alias: "crm", ref: "./crm.ts", path: "/w/crm.ts" },
    { alias: "find", ref: "./crm.ts:lookup", path: "/w/crm.ts" },
    { alias: "gone", ref: "./crm.ts:gone", path: "/w/crm.ts" },
    { alias: "x", ref: "./x.ts", path: "/w/x.ts" },
  ];

  const scope = recordedModuleScope(refs, (path) => recorded.get(path) ?? null);

  expect([[...scope.groups], scope.tools]).toEqual([[["crm", [{ name: "create" }, { name: "lookup" }]]], [{ name: "find" }]]);
  // `find` and `crm`'s `lookup` are one export, bound once.
  const visible = expandGrants(parseToolGrants(["find", "crm"], "n"), [nodeScope(scope)]).map((each) => each.visible);
  expect(visible).toEqual(["find", "crm_create"]);
});

test("a module's key is its path relative to the root flow's folder, or its path with none", () => {
  expect([moduleKey("/w/flows/tools/crm.ts", "/w/flows"), moduleKey("/w/crm.ts", "/w/flows"), moduleKey("/w/crm.ts", null)]).toEqual([
    "tools/crm.ts",
    "../crm.ts",
    "/w/crm.ts",
  ]);
});

/** A thread that answers each request with what it was given; only `lookup`'s calls are safe. */
const thread = {
  check: async (tool: string, name: string, args: unknown) => ({ value: { safe: tool === "lookup", given: [tool, name, args] } }),
  call: async (tool: string, checked: unknown, ctx: ToolContext) => [tool, checked, ctx.node],
  checkOutput: async (tool: string, name: string, value: unknown) => ({ value: [tool, name, value] }),
} as unknown as ToolModuleThread;
const checked = { ...listed("place"), hasParametersValidator: true, hasOutputValidator: true };
const started: StartedToolModule = { path: "/w/crm.ts", key: "crm.ts", thread, listing: { ...LISTING, tools: [...LISTING.tools, checked] } };
const tools = new ToolSet({}).withModules([started]);
const ctx = new ToolContext({ callId: "c", attempt: 1, toolName: "find", runId: "r", node: "a", context: undefined, signal: new AbortController().signal });

test("a node scope holds a group alias's tools by export name and a one-export alias's tool by the alias", async () => {
  const refs = [
    { alias: "crm", ref: "./crm.ts", path: "/w/crm.ts" },
    { alias: "find", ref: "./crm.ts:lookup", path: "/w/crm.ts" },
  ];
  const scope = moduleScope(refs, tools);

  expect([...scope.groups].map(([group, each]) => [group, each.map((tool) => tool.name)])).toEqual([["crm", ["create", "lookup", "place"]]]);
  const [find] = scope.tools;
  expect([find!.name, find!.description, find!.resumable, find!.parametersJsonSchema]).toEqual([
    "find",
    "lookup.",
    true,
    listed("lookup").parametersJsonSchema,
  ]);
  const visible = expandGrants(parseToolGrants(["find", "crm"], "n"), [nodeScope(scope)]).map((each) => each.visible);
  expect(visible).toEqual(["find", "crm_create", "crm_place"]);
  // Each entry's source names its alias, the ref as written and the export.
  expect([scope.sources!.get(JSON.stringify([null, "find"])), scope.sources!.get(JSON.stringify(["crm", "create"]))]).toEqual([
    { kind: "module", alias: "find", ref: "./crm.ts:lookup", export: "lookup" },
    { kind: "module", alias: "crm", ref: "./crm.ts", export: "create" },
  ]);
  // Each stand-in checks its export's arguments on the module's thread, which says whether the call
  // is safe, and calls the export with what the check returned.
  const handle = await find!.parametersValidator!["~standard"].validate({ id: "1" });
  expect(handle).toEqual({ value: { safe: true, given: ["lookup", "find", { id: "1" }] } });
  const { value } = handle as { value: Record<string, unknown> };
  expect([find!.isConcurrencySafe!(value), await find!.execute(value, ctx)]).toEqual([true, ["lookup", value, "a"]]);
  // Only a tool with an output validator has one, which runs on the thread.
  const place = scope.groups.get("crm")!.at(-1)!;
  expect(find!.outputValidator).toBeNull();
  expect(await place.outputValidator!["~standard"].validate({ total: 1 })).toEqual({ value: ["place", "place", { total: 1 }] });
  expect(() => moduleScope([{ alias: "x", ref: "./x.ts", path: "/w/x.ts" }], tools)).toThrow(
    new RangeError("tool module 'x' (./x.ts) wasn't started for this run"),
  );
  expect(() => moduleScope([{ alias: "f", ref: "./crm.ts:find", path: "/w/crm.ts" }], tools)).toThrow(
    new RangeError(LISTING.refused[0]!.message),
  );
});

test("a stand-in sends its node's console with each request, and returns pending() only when the run's host completes calls", async () => {
  const sinks: unknown[] = [];
  const pends = {
    check: async (_tool: string, _name: string, args: unknown, sink: unknown) => {
      sinks.push(sink);
      return { value: { safe: false, args } };
    },
    call: async (_tool: string, _checked: unknown, _ctx: ToolContext, sink: unknown) => {
      sinks.push(sink);
      return pending();
    },
  } as unknown as ToolModuleThread;
  const module: StartedToolModule = { path: "/w/crm.ts", key: "crm.ts", thread: pends, listing: LISTING };
  const refs = [
    { alias: "crm", ref: "./crm.ts", path: "/w/crm.ts" },
    { alias: "find", ref: "./crm.ts:lookup", path: "/w/crm.ts" },
  ];
  const lines = () => {};
  /** What calling `ref` with `tools`' stand-ins returns, or throws. */
  const outcome = (tools: ToolSet, ref: string | { group: string; name: string }) =>
    tools.withNodeScope(moduleScope(refs, tools, lines)).invoke(ref, { id: "1" }).catch((error: Error) => error);

  expect(await outcome(new ToolSet({}).withModules([module], { completesCalls: true }), "find")).toBe(pending());
  expect(sinks).toEqual([lines, lines]);
  const refused = new ToolSet({}).withModules([module]);
  const why = "returned pending(), but this run's host doesn't complete calls";
  for (const [ref, shown] of [["find", "find"], [{ group: "crm", name: "lookup" }, "crm_lookup"]] as const) {
    const error = await outcome(refused, ref);
    expect([isToolCheckError(error), (error as Error).message]).toEqual([true, `tool '${shown}' ${why}`]);
  }
});

/**
 * A tool whose validator and output validator act as `how` says: `issues`, `number` (not a
 * Standard Schema result) and `throw` for the arguments, `out-*` for the result; any other value
 * passes as a class instance, which `isConcurrencySafe` takes as safe for `safe`.
 */
const CHECKED = `const parameters = { type: "object", properties: { how: { type: "string" } }, required: ["how"] };
class Checked {
  constructor(how) { this.how = how; }
}
const validator = (validate) => ({ "~standard": { version: 1, vendor: "t", validate, jsonSchema: { input: () => ({ type: "object" }) } } });
const checker = validator(({ how }) => {
  if (how === "issues") return { issues: [{ message: "too short", path: ["a", 0, { key: "b" }, Symbol("s")] }, { message: "no path" }] };
  if (how === "number") return 5;
  if (how === "throw") throw new TypeError("the validator broke");
  return { value: new Checked(how) };
});
const output = validator((result) => {
  if (result.how === "out-issues") return { issues: [{ message: "wrong", path: ["total"] }] };
  if (result.how === "out-number") return 5;
  if (result.how === "out-throw") throw new RangeError("the output validator broke");
  if (result.how === "out-date") return { value: new Date(0) };
  return { value: { ...result, total: 2.5 } };
});
export const lookup = {
  name: "lookup",
  description: "d",
  parameters: checker,
  parametersJsonSchema: parameters,
  output,
  isConcurrencySafe: (args) => args instanceof Checked && args.how === "safe",
  execute: (args) => ({ how: args.how, checked: args instanceof Checked }),
};
`;

test("a module tool's argument and result checks act as the same tool's do on the host", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "yian-modules-")));
  const path = join(dir, "crm.ts");
  writeFileSync(path, CHECKED);
  const real = new ToolModuleThread(path, "crm.ts");
  try {
    const started: StartedToolModule = { path, key: "crm.ts", thread: real, listing: await real.list() };
    const ref = { alias: "lookup", ref: "./crm.ts:lookup", path };
    const module = new ToolSet({}).withModules([started]);
    const viaModule = module.withNodeScope(moduleScope([ref], module));
    const host = listToolModule(await import(pathToFileURL(path).href), "crm.ts").tools.get("lookup")!;
    const viaHost = new ToolSet({ lookup: host });
    /** What one call does: its error's name, message, kinds and cause's message, or whether it is safe and its result. */
    const outcome = async (tools: ToolSet, inputs: Record<string, unknown>) => {
      try {
        const call = await tools.validate("lookup", inputs);
        return { safe: tools.isConcurrencySafe(call), result: await tools.invoke(call) };
      } catch (error) {
        const { name, message, cause } = error as Error;
        return { error: [name, message, toolErrorKinds(error), (cause as Error | undefined)?.message ?? null] };
      }
    };
    const cases = [{}, ...["ok", "safe", "issues", "number", "throw", "out-issues", "out-number", "out-throw", "out-date"].map((how) => ({ how }))];

    const outcomes = [];
    for (const inputs of cases) outcomes.push(await outcome(viaModule, inputs));

    const expected = [];
    for (const inputs of cases) expected.push(await outcome(viaHost, inputs));
    expect(outcomes).toEqual(expected);
    expect(outcomes).toEqual([
      { error: ["RangeError", 'tool \'lookup\' is missing required argument(s) ["how"]', [], null] },
      { safe: false, result: { how: "ok", checked: true, total: 2.5 } },
      { safe: true, result: { how: "safe", checked: true, total: 2.5 } },
      { error: ["ToolCheckError", "tool 'lookup' arguments failed its validator: a[0].b[\"Symbol(s)\"]: too short; no path", ["check"], null] },
      { error: ["TypeError", "tool 'lookup' arguments: the validator returned number, not a Standard Schema result", [], null] },
      { error: ["TypeError", "the validator broke", [], null] },
      { error: ["ToolCheckError", "tool 'lookup' result failed its output validator: total: wrong", ["check"], null] },
      { error: ["ToolCheckError", "tool 'lookup' result: the output validator returned number, not a Standard Schema result", ["check"], null] },
      { error: ["ToolCheckError", "tool 'lookup' result: the output validator threw: the output validator broke", ["check"], "the output validator broke"] },
      { error: ["ToolCheckError", "tool 'lookup' result: Date is not a yian value", ["check"], "tool 'lookup' result: Date is not a yian value"] },
    ]);
  } finally {
    await real.terminate();
    rmSync(dir, { recursive: true, force: true });
  }
});
