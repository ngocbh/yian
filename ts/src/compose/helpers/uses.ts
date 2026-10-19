/**
 * Pure parsing of a `uses:` ref string into `UsesRef(scheme, path, version)`.
 *
 * Grammar: `[<scheme>:]<path>[@<version>]`.
 *
 * - `<scheme>:`: a leading `[A-Za-z][A-Za-z0-9+.-]*:` token before any `/`, stored lowercased.
 *   `null` is local; `"hub"` is the marketplace; anything else is rejected by the resolver
 *   (this module only extracts the scheme, it does not judge which are known, so a
 *   Windows-drive-looking `C:/x` parses to scheme `c` and never touches the disk).
 * - `<path>`: the (relative) flow path, possibly nested. Required.
 * - `@<version>`: an optional opaque tag, split at the last `@` (`a@b@c` gives path `a@b`,
 *   version `c`); a trailing `@` (empty version) is an error.
 *
 * Imports only `compose/helpers/errors` (the lowest layer, so the loader and the resolver can
 * both import this without a cycle).
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
  const raw = (value || "").trim();
  if (!raw) throw diagnostics.error("uses: ref must be a non-empty string");

  let scheme: string | null = null;
  let rest = raw;
  // The regex anchors at the start and excludes `/`, so its colon always precedes any `/`.
  const match = SCHEME.exec(raw);
  if (match !== null) {
    scheme = match[1]!.toLowerCase();
    rest = raw.slice(match[0].length);
  }

  const at = rest.lastIndexOf("@");
  const path = at === -1 ? rest : rest.slice(0, at);
  const version = at === -1 ? null : rest.slice(at + 1);

  if (!path) throw diagnostics.error(`uses: ref ${reprText(value)} has no flow path`);
  if (version === "") {
    throw diagnostics.error(`uses: ref ${reprText(value)} has an empty version (trailing '@')`);
  }
  return new UsesRef(scheme, path, version);
}
