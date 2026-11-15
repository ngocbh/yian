/**
 * Which exports of a tool module are tools, as its thread reads them: plain objects of `tool()`'s
 * options and `tool()` values from any yian copy, each named as its export, in export-name order.
 * A refused or unreadable export is reported for itself, and anything else is ignored. The
 * modules are imported on this thread.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { listToolModule, type ToolModuleListing } from "../../src/tools/module_worker.ts";

const SOURCE = join(import.meta.dirname, "..", "..", "src");
const PARAMS = 'parameters: { type: "object" }';

let tmp: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-module-worker-")));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Write `source` as `file` in `tmp` and return its namespace. */
async function imported(source: string, file = "crm.ts"): Promise<object> {
  writeFileSync(join(tmp, file), source);
  return import(pathToFileURL(join(tmp, file)).href) as Promise<object>;
}

/** The source of a plain-object tool exported and named as `name`. */
function plain(name: string): string {
  return `export const ${name} = { name: ${JSON.stringify(name)}, description: "d", ${PARAMS}, execute: () => 1 };\n`;
}

test("plain objects with an execute function are tools, in export-name order; other exports are ignored", async () => {
  const namespace = await imported(
    ["zeta", "alpha", "Beta", "$dollar", "_under"].map(plain).join("") +
      `export const bare = Object.assign(Object.create(null), { name: "bare", description: "d", ${PARAMS}, execute: () => 1 });
export default { name: "default", description: "d", ${PARAMS}, execute: () => 1 };
export const LIMIT = 10;
export class Tool { execute() {} }
export const instance = new Tool();
export function execute() {}
export const list = [() => 1];
export const none = null;
export const config = { name: "config", execute: "not a function" };
`,
  );

  const { listing } = listToolModule(namespace, "crm.ts");

  expect(listing.tools.map((tool) => tool.name)).toEqual(["$dollar", "Beta", "_under", "alpha", "bare", "zeta"]);
  expect(listing.refused).toEqual([]);
  expect(listing.others).toEqual(["LIMIT", "Tool", "config", "default", "execute", "instance", "list", "none"]);
});

test("a listed tool carries its description, both schemas, resumable and which functions it has", async () => {
  const namespace = await imported(`const checker = {
  "~standard": { version: 1, vendor: "t", validate: (value) => ({ value }), jsonSchema: { input: () => ({ type: "object" }) } },
};
export const lookup = {
  name: "lookup",
  description: "Find a customer.",
  parameters: { type: "object", properties: { id: { type: "string" } } },
  output: { type: "object", properties: { found: { type: "boolean" } } },
  isConcurrencySafe: () => true,
  resumable: true,
  execute: () => ({ found: true }),
};
export const place = { name: "place", description: "Place it.", parameters: checker, output: checker, execute: () => ({}) };
`);

  const { listing, tools } = listToolModule(namespace, "crm.ts");

  expect(listing.tools).toEqual([
    {
      name: "lookup",
      description: "Find a customer.",
      parametersJsonSchema: { type: "object", properties: { id: { type: "string" } } },
      outputJsonSchema: { type: "object", properties: { found: { type: "boolean" } } },
      resumable: true,
      hasParametersValidator: false,
      hasOutputValidator: false,
      hasIsConcurrencySafe: true,
    },
    {
      name: "place",
      description: "Place it.",
      parametersJsonSchema: { type: "object" },
      outputJsonSchema: { type: "object" },
      resumable: false,
      hasParametersValidator: true,
      hasOutputValidator: true,
      hasIsConcurrencySafe: false,
    },
  ]);
  expect([...tools.keys()]).toEqual(["lookup", "place"]);
});

test("a plain object's execute and isConcurrencySafe run with the module's own object as this", async () => {
  const namespace = await imported(`export const lookup = {
  name: "lookup",
  description: "d",
  ${PARAMS},
  isConcurrencySafe(args) { return this === lookup && args.id === "1"; },
  execute(args) { return this === lookup ? [this.name, args.id] : "unbound"; },
};
`);

  const lookup = listToolModule(namespace, "crm.ts").tools.get("lookup")!;

  expect([lookup.execute({ id: "1" }, {} as never), lookup.isConcurrencySafe!({ id: "1" })]).toEqual([["lookup", "1"], true]);
});

test("a tool() value from any yian copy is copied, a field an older copy lacks read as absent", async () => {
  const functionTools = pathToFileURL(join(SOURCE, "tools", "function_tools.ts")).href;
  const namespace = await imported(`import { tool } from ${JSON.stringify(functionTools)};
export const made = tool({ name: "made", description: "d", ${PARAMS}, resumable: true, execute: () => 1 });
export const renamed = tool({ name: "other", description: "d", ${PARAMS}, execute: () => 1 });
// An older copy's tool: the brand and a tool's shape, without the fields added since.
const brand = Symbol.for("yian.function_tool");
export const older = { [brand]: true, name: "older", description: "d", parametersJsonSchema: { type: "object" }, execute: () => 1 };
`);

  const { listing, tools } = listToolModule(namespace, "crm.ts");

  expect(listing.tools.map((tool) => [tool.name, tool.resumable, tool.outputJsonSchema])).toEqual([
    ["made", true, null],
    ["older", false, null],
  ]);
  expect(tools.get("made")).not.toBe((namespace as { made: unknown }).made);
  expect(listing.refused).toEqual([
    { name: "renamed", message: "crm.ts export 'renamed' holds a tool named 'other'; rename one" },
  ]);
});

test.each([
  [`description: "d", ${PARAMS}`, "has no 'name'"],
  [`name: "lookup", ${PARAMS}`, "has no 'description'"],
  ['name: "lookup", description: "d"', "has no 'parameters'"],
  [`name: 5, description: "d", ${PARAMS}`, "name must be a string, got number"],
  [`name: "find", description: "d", ${PARAMS}`, "holds a tool named 'find'; rename one"],
  [`name: "lookup", description: 5, ${PARAMS}`, "description must be a string, got number"],
  ['name: "lookup", description: "d", parameters: "id"', "parameters must be a JSON Schema object or a Standard Schema v1 validator"],
  [`name: "lookup", description: "d", ${PARAMS}, resumable: "yes"`, "resumable must be a boolean, got string"],
  [`name: "lookup", description: "d", ${PARAMS}, isConcurrencySafe: 1`, "isConcurrencySafe must be a function"],
  [`name: "lookup", description: "d", ${PARAMS}, resumeable: true`, "has an unknown key 'resumeable'; did you mean 'resumable'?"],
  [`name: "lookup", description: "d", ${PARAMS}, zzz: 1`, "has an unknown key 'zzz'"],
])("a plain object { %s } is refused, naming the export and the field", async (fields, why) => {
  const namespace = await imported(`export const lookup = { ${fields}, execute: () => 1 };\n${plain("ok")}`);

  const { listing } = listToolModule(namespace, "crm.ts");

  expect(listing.refused).toEqual([{ name: "lookup", message: `crm.ts export 'lookup' ${why}` }]);
  expect(listing.tools.map((tool) => tool.name)).toEqual(["ok"]);
});

test("an export behind a throwing getter or a throwing trap is reported for itself", async () => {
  const namespace = await imported(`export const sly = { get execute() { throw new RangeError("no execute"); } };
export const trap = new Proxy({}, { getPrototypeOf() { throw new Error("trapped"); } });
${plain("ok")}`);

  const { listing } = listToolModule(namespace, "crm.ts");

  expect(listing.refused).toEqual([
    { name: "sly", message: "crm.ts export 'sly' can't be read: RangeError: no execute" },
    { name: "trap", message: "crm.ts export 'trap' can't be read: Error: trapped" },
  ]);
  expect(listing.tools.map((tool) => tool.name)).toEqual(["ok"]);
});

test("an export in its TDZ is reported for itself, and the listing still enumerates it", async () => {
  // `seen.ts` lists `cycle.ts` while `cycle.ts` waits on it, so its exports aren't initialized yet.
  const moduleWorker = pathToFileURL(join(SOURCE, "tools", "module_worker.ts")).href;
  writeFileSync(
    join(tmp, "seen.ts"),
    `import * as cycle from "./cycle.ts";
import { listToolModule } from ${JSON.stringify(moduleWorker)};
export const seen = listToolModule(cycle, "cycle.ts").listing;
`,
  );
  await imported(`import "./seen.ts";\n${plain("early")}export const LIMIT = 1;\n`, "cycle.ts");
  const { seen } = (await import(pathToFileURL(join(tmp, "seen.ts")).href)) as { seen: ToolModuleListing };

  expect([seen.tools, seen.others]).toEqual([[], []]);
  expect(seen.refused.map((each) => each.name)).toEqual(["LIMIT", "early"]);
  for (const { name, message } of seen.refused) {
    expect(message).toStartWith(`cycle.ts export '${name}' can't be read: ReferenceError: `);
  }
});
