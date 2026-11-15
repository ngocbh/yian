/**
 * Tool modules: TypeScript or JavaScript files a flow imports through `uses:`, whose exports are
 * tools. An alias names every tool a module exports (`crm: ./crm.ts`), a group, or one export
 * (`quote: ./tools.ts:quote`), one tool. A load never imports a module.
 *
 * Knows about: typesys.
 */

import { YianTypeError } from "../typesys/errors.ts";
import { isRecord } from "../typesys/values.ts";

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
