/**
 * A started tool module's checks and node scope: what a run refuses in a listing for the aliases
 * that reach it, and the stand-ins a node's scope holds for its tools, each keyed by its file and
 * export so two aliases of one file bind a tool once, and each calling its export on the module's
 * thread.
 */

import { expect, test } from "bun:test";

import { ToolContext } from "../../src/tools/context.ts";
import { PermanentToolError } from "../../src/tools/errors.ts";
import { nodeScope, expandGrants, parseToolGrants } from "../../src/tools/grants.ts";
import type { ToolModuleThread } from "../../src/tools/module_thread.ts";
import type { ListedTool, ToolModuleListing } from "../../src/tools/module_worker.ts";
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

/** A thread that answers each call with what it was given: the export, the arguments and the node. */
const thread = { call: async (tool: string, args: unknown, ctx: ToolContext) => [tool, args, ctx.node] } as unknown as ToolModuleThread;
const checked = { ...listed("place"), hasParametersValidator: true };
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
  // Each stand-in calls its export on the module's thread.
  expect(await find!.execute({ id: "1" }, ctx)).toEqual(["lookup", { id: "1" }, "a"]);
  const place = scope.groups.get("crm")!.at(-1)!;
  expect(() => place.execute({}, ctx)).toThrow(new PermanentToolError("crm.ts export 'place': a tool module's validators aren't supported yet"));
  expect(() => moduleScope([{ alias: "x", ref: "./x.ts", path: "/w/x.ts" }], tools)).toThrow(
    new RangeError("tool module 'x' (./x.ts) wasn't started for this run"),
  );
  expect(() => moduleScope([{ alias: "f", ref: "./crm.ts:find", path: "/w/crm.ts" }], tools)).toThrow(
    new RangeError(LISTING.refused[0]!.message),
  );
});
