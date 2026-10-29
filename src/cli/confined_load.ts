/**
 * Loading a flow confined to a root folder, as `yian check --root` does and `yian chat`'s
 * assistant does with the flows it reads and writes: the flow, its `uses:` children and its
 * `./file.ts:fn` CODE references must all be inside the root.
 *
 * Knows about: compose (the loader and the source parser), the platform's path helpers and the
 * file system.
 */

import { realpathSync } from "node:fs";
import { dirname, join, relative } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import { loadFlow } from "../compose/loader.ts";
import { parseSourceDocument } from "../compose/parse/file.ts";
import type { Env } from "../defaults.ts";
import { Diagnostic, reprText } from "../diagnostics.ts";
import { isSystemError, leavesRoot, statOrNull } from "../platform.ts";
import { resolvedPath } from "../skills/catalog.ts";
import { isRecord } from "../typesys/values.ts";
import { readFlowText } from "./startup.ts";

/**
 * Return `path` resolved without requiring it to exist (the real path of its longest existing
 * prefix, with the rest appended), except that a symlink loop throws its `ELOOP` system error.
 */
export function resolveFlowPath(path: string): string {
  try {
    return realpathSync(path);
  } catch (error) {
    if (!isSystemError(error) || error.code === "ELOOP") throw error;
    return resolvedPath(path);
  }
}

/**
 * Load the flow at `path` from inside `root`, labelled with its path relative to `root`; its
 * `uses:` children load the same way, each ref only from `<importing file's dir>/<ref>.yaml`.
 * Every `./file.ts:fn` CODE reference must resolve inside `root` too.
 *
 * The loader puts the `uses:` alias frame on a child's `LoadError` and turns anything else a child
 * throws into a `compose.load` error there.
 *
 * @param root The resolved root directory.
 * @param loading The resolved paths of the flows importing this one.
 * @throws {LoadError} `compose.outside_root` for a flow outside `root`; `compose.load` for a flow
 *   whose YAML doesn't parse, that sets `system.paths`, or that doesn't load.
 * @throws {Error} A reference cycle.
 * @throws {NotUtf8Error} When the flow isn't UTF-8 text.
 * @throws {Error} A system error when the flow can't be read.
 */
export async function loadConfined(
  root: string,
  path: string,
  loading: ReadonlySet<string>,
  env: Env,
): Promise<ComposedFlow> {
  return loadConfinedSource(root, path, null, loading, env);
}

/**
 * Load `text`, a flow not on disk, as `loadConfined` would load it from `virtualPath`: the path
 * labels its diagnostics and its folder is where its `uses:` children and CODE references
 * resolve from.
 *
 * @param root The resolved root directory.
 * @throws As `loadConfined` throws, but never for reading the flow itself.
 */
export async function loadConfinedText(
  root: string,
  virtualPath: string,
  text: string,
  env: Env,
): Promise<ComposedFlow> {
  return loadConfinedSource(root, virtualPath, text.replace(/\r\n?/g, "\n"), new Set(), env);
}

/** Load the flow at `path`, from `text` when given, else read from the file, as `loadConfined` describes. */
async function loadConfinedSource(
  root: string,
  path: string,
  given: string | null,
  loading: ReadonlySet<string>,
  env: Env,
): Promise<ComposedFlow> {
  // Resolved paths are compared, so a symlinked root (/var -> /private/var on macOS) works.
  const resolved = resolveFlowPath(path);
  const shown = relative(root, resolved) || ".";
  if (leavesRoot(shown)) {
    const message = `${shown} is outside the root`;
    throw new LoadError(new Diagnostic({ code: "compose.outside_root", message }));
  }
  if (loading.has(resolved)) throw new Error(`flow-reference cycle at ${shown}`);
  const text = given ?? readFlowText(resolved);
  const doc = parseSourceDocument(text, shown).root.value;
  // The loader doesn't search `system.paths` when given a child resolver, so such a flow would
  // load differently from `yian run`.
  if (isRecord(doc) && isRecord(doc["system"]) && Object.hasOwn(doc["system"], "paths")) {
    throw new LoadError(
      new Diagnostic({
        code: "compose.load",
        message: "system.paths is not supported with --root; use a relative uses: path",
      }),
    );
  }
  const dir = dirname(resolved);
  const importing = new Set([...loading, resolved]);
  // Unlike the default file resolver, this keeps no reuse cache, so a child imported along several
  // paths loads once per path; that is fine for a check.
  const childResolver = async (ref: string, version: string | null): Promise<ComposedFlow> => {
    const childPath = join(dir, `${ref}.yaml`);
    // A missing child outside the root is reported as outside, by `loadConfinedSource`.
    const inside = !leavesRoot(relative(root, resolveFlowPath(childPath)));
    if (inside && !statOrNull(childPath)?.isFile()) {
      throw new Error(`flow ${reprText(ref)} not found: ${relative(root, childPath)}`);
    }
    // The loader holds the child to a pinned `@<version>`.
    return loadConfinedSource(root, childPath, null, importing, env);
  };
  // `CodeNode.check()` resolves a `./file.ts:fn` reference against `baseDir`, here and in a child,
  // and refuses one outside `codeRoot`; a child is loaded here, so it gets the root too.
  return loadFlow(text, { childResolver, sourceLabel: shown, baseDir: dir, codeRoot: root, env });
}
