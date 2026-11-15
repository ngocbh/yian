/**
 * Tool modules: TypeScript or JavaScript files a flow imports through `uses:`, whose exports are
 * tools. An alias names every tool a module exports (`crm: ./crm.ts`), a group, or one export
 * (`quote: ./tools.ts:quote`), one tool. A load never imports a module; each segment of a run
 * starts the modules its nodes reach, and a node's scope holds a stand-in for each tool it reaches.
 *
 * Knows about: tools/errors, tools/function_tools, the types of tools/grants, tools/module_thread,
 * tools/module_worker and tools/registry, diagnostics and typesys.
 */

import { getCloseMatches, reprText } from "../diagnostics.ts";
import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import { isRecord } from "../typesys/values.ts";
import { PermanentToolError } from "./errors.ts";
import { FunctionTool } from "./function_tools.ts";
import type { NodeScope } from "./grants.ts";
import type { ToolModuleThread } from "./module_thread.ts";
import type { ListedTool, ToolModuleListing } from "./module_worker.ts";
import type { ToolSet } from "./registry.ts";

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
 * each a stand-in with its listed spec. Every entry's identity is its file and export, so two
 * aliases of one file bind a tool once.
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
    const { key, listing } = started;
    const identity = (listed: ListedTool) => JSON.stringify({ module: path, export: listed.name });
    const { exportName } = splitModuleRef(ref);
    if (exportName === null) {
      groups.set(alias, listing.tools.map((listed) => standIn(key, listed, listed.name)));
      for (const listed of listing.tools) identities.set(JSON.stringify([alias, listed.name]), identity(listed));
    } else {
      const listed = listedExport(key, listing, exportName);
      single.push(standIn(key, listed, alias));
      identities.set(JSON.stringify([null, alias]), identity(listed));
    }
  }
  return { groups, tools: single, identities };
}

/** Return the engine's stand-in for a module's listed tool, named `name`, with its listed spec. */
function standIn(key: string, listed: ListedTool, name: string): FunctionTool {
  return new FunctionTool({
    name,
    description: listed.description,
    parametersJsonSchema: listed.parametersJsonSchema,
    outputJsonSchema: listed.outputJsonSchema,
    resumable: listed.resumable,
    execute: () => {
      const message = `${key} export ${reprText(listed.name)}: calls to tool modules aren't supported yet`;
      throw new PermanentToolError(message);
    },
  });
}
