/**
 * A started tool module's node scope: the stand-ins a node's scope holds for its tools, each keyed
 * by its file and export so two aliases of one file bind a tool once.
 */

import { expect, test } from "bun:test";

import { nodeScope, expandGrants, parseToolGrants } from "../../src/tools/grants.ts";
import type { ToolModuleThread } from "../../src/tools/module_thread.ts";
import type { ListedTool, ToolModuleListing } from "../../src/tools/module_worker.ts";
import { moduleScope, type StartedToolModule } from "../../src/tools/modules.ts";
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

const started: StartedToolModule = { path: "/w/crm.ts", key: "crm.ts", thread: {} as ToolModuleThread, listing: LISTING };
const tools = new ToolSet({}).withModules([started]);

test("a node scope holds a group alias's tools by export name and a one-export alias's tool by the alias", () => {
  const refs = [
    { alias: "crm", ref: "./crm.ts", path: "/w/crm.ts" },
    { alias: "find", ref: "./crm.ts:lookup", path: "/w/crm.ts" },
  ];
  const scope = moduleScope(refs, tools);

  expect([...scope.groups].map(([group, each]) => [group, each.map((tool) => tool.name)])).toEqual([["crm", ["create", "lookup"]]]);
  const [find] = scope.tools;
  expect([find!.name, find!.description, find!.resumable, find!.parametersJsonSchema]).toEqual([
    "find",
    "lookup.",
    true,
    listed("lookup").parametersJsonSchema,
  ]);
  const visible = expandGrants(parseToolGrants(["find", "crm"], "n"), [nodeScope(scope)]).map((each) => each.visible);
  expect(visible).toEqual(["find", "crm_create"]);
  expect(() => find!.execute({}, {} as never)).toThrow("crm.ts export 'lookup': calls to tool modules aren't supported yet");
  expect(() => moduleScope([{ alias: "x", ref: "./x.ts", path: "/w/x.ts" }], tools)).toThrow(
    new RangeError("tool module 'x' (./x.ts) wasn't started for this run"),
  );
  expect(() => moduleScope([{ alias: "f", ref: "./crm.ts:find", path: "/w/crm.ts" }], tools)).toThrow(
    new RangeError(LISTING.refused[0]!.message),
  );
});
