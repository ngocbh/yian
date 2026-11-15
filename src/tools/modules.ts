/**
 * Tool modules: TypeScript or JavaScript files a flow imports through `uses:`, whose exports are
 * tools. An alias names every tool a module exports (`crm: ./crm.ts`), a group, or one export
 * (`quote: ./tools.ts:quote`), one tool. A load never imports a module; each segment of a run
 * starts the modules its nodes reach, and a node's scope holds a stand-in for each tool it reaches,
 * which the registry checks and calls as a host tool, and whose call runs on the module's thread.
 *
 * Knows about: tools/function_tools, tools/registry, the types of tools/grants,
 * tools/module_thread and tools/module_worker, diagnostics, typesys and `node:path`.
 */

import { relative } from "node:path";

import { getCloseMatches, reprText } from "../diagnostics.ts";
import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import { isRecord } from "../typesys/values.ts";
import { FunctionTool, type StandardSchemaV1 } from "./function_tools.ts";
import type { NodeScope } from "./grants.ts";
import type { ToolModuleThread } from "./module_thread.ts";
import type { ListedTool, ToolModuleListing } from "./module_worker.ts";
import { RESERVED_TOOL_NAMES, type ToolSet } from "./registry.ts";

/** A tool module an AGENT grants or a TOOL node calls, as the loader resolved its `uses:` alias. Frozen. */
export interface ToolModuleRef {
  /** The `uses:` alias. */
  readonly alias: string;
  /** The alias's `uses:` value as written: `./crm.ts`, or `./tools.ts:quote` for one export. */
  readonly ref: string;
  /** The module file's real path. */
  readonly path: string;
}

/**
 * Split a tool module ref into its path and the export it names: `./tools.ts:quote` into
 * `./tools.ts` and `quote`, `./crm.ts` into `./crm.ts` and `null`, which names every export.
 */
export function splitModuleRef(ref: string): { readonly path: string; readonly exportName: string | null } {
  // A module's path holds no colon, as a CODE reference's doesn't.
  const [path, exportName = null] = ref.trim().split(/:(.*)/s);
  return { path: path!, exportName };
}

/**
 * Return a frozen copy of `module`, checked.
 *
 * @param where The node, for the message.
 * @throws TypeError when it isn't a record of a non-empty `alias`, `ref` and `path`.
 */
export function checkedToolModule(where: string, module: ToolModuleRef): ToolModuleRef {
  const { alias, ref, path } = isRecord(module) ? module : ({} as Partial<ToolModuleRef>);
  if (![alias, ref, path].every((field) => typeof field === "string" && field !== "")) {
    throw new YianTypeError(`${where}: a tool module needs a non-empty alias, ref and path`);
  }
  return Object.freeze({ alias: alias!, ref: ref!, path: path! });
}

/** A tool module one segment of a run started: its file, its thread, and the tools it listed. */
export interface StartedToolModule {
  /** The module file's real path, which the run holds it by. */
  readonly path: string;
  /** How messages name the file: its path relative to the root flow's folder. */
  readonly key: string;
  readonly thread: ToolModuleThread;
  readonly listing: ToolModuleListing;
}

/**
 * Return a module's key: its real `path` relative to `dir`, the real folder of the run's root
 * flow, or `path` itself for a flow with no folder.
 */
export function moduleKey(path: string, dir: string | null): string {
  return dir === null ? path : relative(dir, path);
}

/**
 * Refuse a module's listing for what reaches it: each export a one-export alias names must be a
 * tool; with `group`, an alias names every tool it exports, so none may be refused or take a name
 * yian reserves, and there must be one.
 *
 * @param key How messages name the file.
 * @throws RangeError naming the file and the export.
 */
export function checkModuleListing(
  key: string,
  listing: ToolModuleListing,
  reach: { readonly group: boolean; readonly exports: readonly string[] },
): void {
  for (const name of reach.exports) listedExport(key, listing, name);
  if (!reach.group) return;
  const [refused] = listing.refused;
  if (refused !== undefined) throw new YianRangeError(refused.message);
  if (listing.tools.length === 0) throw new YianRangeError(`${key} exports no tools`);
  const reserved = listing.tools.find(({ name }) => RESERVED_TOOL_NAMES.has(name));
  if (reserved !== undefined) {
    throw new YianRangeError(`${key} export ${reprText(reserved.name)} takes a name yian reserves; rename it`);
  }
}

/**
 * Return export `name` of a module's listing, as a tool.
 *
 * @throws RangeError naming the file and the export when it is refused, isn't a tool, or is missing.
 */
function listedExport(key: string, listing: ToolModuleListing, name: string): ListedTool {
  const found = listing.tools.find((tool) => tool.name === name);
  if (found !== undefined) return found;
  const refused = listing.refused.find((each) => each.name === name);
  if (refused !== undefined) throw new YianRangeError(refused.message);
  const head = `${key} export ${reprText(name)}`;
  if (listing.others.includes(name)) {
    throw new YianRangeError(`${head} isn't a tool: export an object with name, description, parameters and execute`);
  }
  const [close] = getCloseMatches(name, listing.tools.map((tool) => tool.name), 1);
  const hint = close === undefined ? "" : `; did you mean ${reprText(close)}?`;
  throw new YianRangeError(`${key} has no export ${reprText(name)}${hint}`);
}

/**
 * Return the node scope `refs` reach in the modules `tools` started: a group alias's tools as a
 * group under the alias, each by its export name, and a one-export alias's tool under the alias,
 * each a stand-in with its listed spec that calls its tool on the module's thread. Every entry's
 * identity is its file and export, so two aliases of one file bind a tool once.
 *
 * @throws RangeError when a ref's module wasn't started, or its export is refused, isn't a tool or
 *   is missing.
 */
export function moduleScope(refs: readonly ToolModuleRef[], tools: ToolSet): NodeScope<FunctionTool> {
  const groups = new Map<string, FunctionTool[]>();
  const single: FunctionTool[] = [];
  const identities = new Map<string, string>();
  for (const { alias, ref, path } of refs) {
    const started = tools.toolModule(path);
    if (started === null) {
      throw new YianRangeError(`tool module ${reprText(alias)} (${ref}) wasn't started for this run`);
    }
    const { key, thread, listing } = started;
    const identity = (listed: ListedTool) => JSON.stringify({ module: path, export: listed.name });
    const { exportName } = splitModuleRef(ref);
    if (exportName === null) {
      groups.set(alias, listing.tools.map((listed) => standIn(thread, listed, listed.name)));
      for (const listed of listing.tools) identities.set(JSON.stringify([alias, listed.name]), identity(listed));
    } else {
      const listed = listedExport(key, listing, exportName);
      single.push(standIn(thread, listed, alias));
      identities.set(JSON.stringify([null, alias]), identity(listed));
    }
  }
  return { groups, tools: single, identities };
}

/**
 * Return the engine's stand-in for a module's listed tool, named `name`, with its listed spec.
 * The registry runs its checks as a host tool's, and each of the module's functions runs on the
 * module's thread: its validator checks a call's arguments there (with the module's validator, if
 * any, and `isConcurrencySafe`), returning a handle to them that `isConcurrencySafe` reads and
 * `execute` calls the tool with; a tool with an output validator gets one that runs the module's.
 */
function standIn(thread: ToolModuleThread, listed: ListedTool, name: string): FunctionTool {
  const validator = (validate: StandardSchemaV1["~standard"]["validate"]): StandardSchemaV1 => ({
    "~standard": { version: 1, vendor: "yian", validate },
  });
  return new FunctionTool({
    name,
    description: listed.description,
    parametersJsonSchema: listed.parametersJsonSchema,
    outputJsonSchema: listed.outputJsonSchema,
    resumable: listed.resumable,
    parametersValidator: validator((args) => thread.check(listed.name, name, args)),
    isConcurrencySafe: (checked) => checked.safe === true,
    execute: (checked, ctx) => thread.call(listed.name, checked, ctx),
    outputValidator: listed.hasOutputValidator
      ? validator((value) => thread.checkOutput(listed.name, name, value))
      : null,
  });
}

/**
 * Return the node scope `refs` reach in the tools a run recorded of its modules, by alias, as
 * `moduleScope` builds it from their listings: a group alias's recorded tools as a group under the
 * alias, and a one-export alias's tool under the alias, each entry's identity its file and export.
 * `recorded` gives a module's recorded tools by its real path, or `null` for none; a ref whose
 * module or export the run didn't record reaches nothing.
 */
export function recordedModuleScope<T extends { readonly name: string }>(
  refs: readonly ToolModuleRef[],
  recorded: (path: string) => readonly T[] | null,
): NodeScope<T> {
  const groups = new Map<string, T[]>();
  const single: T[] = [];
  const identities = new Map<string, string>();
  for (const { alias, ref, path } of refs) {
    const tools = recorded(path);
    if (tools === null) continue;
    const identity = (name: string) => JSON.stringify({ module: path, export: name });
    const { exportName } = splitModuleRef(ref);
    if (exportName === null) {
      groups.set(alias, [...tools]);
      for (const { name } of tools) identities.set(JSON.stringify([alias, name]), identity(name));
      continue;
    }
    const tool = tools.find(({ name }) => name === exportName);
    if (tool === undefined) continue;
    single.push({ ...tool, name: alias });
    identities.set(JSON.stringify([null, alias]), identity(exportName));
  }
  return { groups, tools: single, identities };
}
