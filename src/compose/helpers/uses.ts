/**
 * Pure parsing of a `uses:` ref string into `UsesRef(scheme, path, version)`, or into the parts of
 * a tool module ref (`moduleRefParts`), which is read first.
 *
 * Grammar: `[<scheme>:]<path>[@<version>]`.
 *
 * - `<scheme>:`: a leading `[A-Za-z][A-Za-z0-9+.-]*:` token before any `/`, stored lowercased.
 *   `null` is local; `"hub"` is the marketplace; anything else is rejected by the resolver
 *   (this module only extracts the scheme, it does not judge which are known, so a
 *   Windows-drive-looking `C:/x` parses to scheme `c` and never touches the disk).
 * - `<path>`: the (relative) flow path, possibly nested. Required.
 * - `@<version>`: an optional opaque tag, split at the last `@` of the last path segment
 *   (`a@b@c` gives path `a@b`, version `c`; the `@` in `./vendor/@acme/flow.yaml` is part of
 *   the path); a trailing `@` (empty version) is an error.
 *
 * Imports only `diagnostics` and `compose/helpers/errors` (the lowest layers, so the loader
 * and the resolver can both import this without a cycle).
 */

import { reprText } from "../../diagnostics.ts";
import { LoadDiagnostics } from "./errors.ts";

const SCHEME = /^([A-Za-z][A-Za-z0-9+.-]*):/;

/** A parsed `uses:` ref; frozen. */
export class UsesRef {
  /** `null` = local; `"hub"` = marketplace; else rejected at resolve. */
  readonly scheme: string | null;
  /** The relative flow path, possibly nested. */
  readonly path: string;
  /** The opaque version tag, if any. */
  readonly version: string | null;

  constructor(scheme: string | null, path: string, version: string | null) {
    this.scheme = scheme;
    this.path = path;
    this.version = version;
    Object.freeze(this);
  }
}

/**
 * Parse a `uses:` value into a `UsesRef`.
 *
 * @throws LoadError on a malformed ref: an empty one, one with no flow path, or one with an
 *   empty version.
 */
export function parseUsesRef(
  value: string,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): UsesRef {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const raw = value.trim();
  if (!raw) throw diagnostics.error("uses: ref must be a non-empty string");

  let scheme: string | null = null;
  let rest = raw;
  // The regex anchors at the start and excludes `/`, so its colon always precedes any `/`.
  const match = SCHEME.exec(raw);
  if (match !== null) {
    scheme = match[1]!.toLowerCase();
    rest = raw.slice(match[0].length);
  }

  const [path, version] = splitVersion(rest);
  if (!path) throw diagnostics.error(`uses: ref ${reprText(value)} has no flow path`);
  if (version === "") {
    throw diagnostics.error(`uses: ref ${reprText(value)} has an empty version (trailing '@')`);
  }
  return new UsesRef(scheme, path, version);
}

/** Split `ref` at its `@<version>`: the last `@` of its last path segment. */
function splitVersion(ref: string): [string, string | null] {
  // An `@` in a directory (an npm-style `@scope/`) is part of the path, not a version.
  const last = ref.lastIndexOf("@");
  const at = last > ref.lastIndexOf("/") ? last : -1;
  return at === -1 ? [ref, null] : [ref.slice(0, at), ref.slice(at + 1)];
}

// A path with no whitespace or colon, as a CODE reference's, ending in a JS or TS extension; then
// an optional `:` and an export name.
const MODULE_REF = /^([^\s:]+\.(?:ts|js|mjs))(?::(.*))?$/;

/** The parts of a `uses:` value that names a tool module: `./tools.ts:quote@2`. */
export interface ModuleRefParts {
  /** The module's path, as written: `./tools.ts`. */
  readonly path: string;
  /** The export after the path's `:`, as written; `null` for none, which names every export. */
  readonly exportName: string | null;
  /** The `@<version>`, which a module can't have; `null` for none. */
  readonly version: string | null;
}

/**
 * Read `value` as a tool module ref, a path ending in `.ts`, `.js` or `.mjs` and an optional
 * `:<export>`, once any `@<version>` is set aside; `null` when it isn't one. Read before
 * `parseUsesRef`, which would take `tools.ts:` in `tools.ts:quote` for a scheme.
 */
export function moduleRefParts(value: string): ModuleRefParts | null {
  const [rest, version] = splitVersion(value.trim());
  const match = MODULE_REF.exec(rest);
  if (match === null) return null;
  return Object.freeze({ path: match[1]!, exportName: match[2] ?? null, version });
}
