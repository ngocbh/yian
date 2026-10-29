/**
 * Occurrence-backed origin resolution for the Compose front end.
 *
 * A source catalog is the format-specific authority behind source-neutral `OriginRef` values.
 * It retains a physical `SourceDocument` occurrence and replays that occurrence only when a
 * diagnostic needs a concrete frame. No source text is parsed or copied during resolution.
 *
 * The mutable builder belongs to Compose construction. `freeze` takes an immutable snapshot so
 * later registrations can't change an already returned catalog.
 */

import {
  DiagnosticFrame,
  OriginRef,
  type OriginResolver,
  reprText,
  type SourceRange,
} from "../../diagnostics.ts";
import { jsTypeName } from "../../typesys/index.ts";
import { type OccurrencePath, SourceAccessError } from "./common.ts";
import { SourceCursor } from "./cursor.ts";
import { SourceDocument } from "./document.ts";

/** A deterministic semantic path local to one Compose catalog. */
export type SourceSitePath = readonly (string | number)[];

/** The range a site selects within its occurrence: the value's or the key's. */
export type SiteAnchor = "value" | "key";

/** @internal One zero-based line selector within a constructed scalar value. */
export interface ConstructedLine {
  readonly index: number;
}

/**
 * @internal One physical YAML occurrence and the range selected within it: the single-parse
 * document that owns it, the path replayed from the document root, and a whole-occurrence
 * anchor or a constructed-line selector.
 */
export interface SourceSite {
  readonly document: SourceDocument;
  readonly occurrencePath: OccurrencePath;
  readonly selector: SiteAnchor | ConstructedLine;
}

// Keeps rebased keys apart from document-local site keys, which always begin with `/`.
const SCOPED_ORIGIN_PREFIX = "@scope";

/**
 * Encode exact text and integer segments as one opaque, non-empty origin key: a typed
 * JSON-Pointer-style key, where `~` and `/` in text are escaped as JSON Pointer does.
 *
 * Example: `encodeSitePath(["nodes", 1, "a/b"])` is `"/s:nodes/i:1/s:a~1b"`.
 *
 * @throws TypeError when `path` isn't an array or a segment is neither text nor an integer.
 * @throws RangeError when `path` is empty.
 */
export function encodeSitePath(path: SourceSitePath): string {
  if (!Array.isArray(path)) throw new TypeError("source site path must be an array");
  if (path.length === 0) throw new RangeError("source site path cannot be empty");
  const encoded: string[] = [];
  for (const segment of path as readonly unknown[]) {
    if (typeof segment === "string") {
      encoded.push("s:" + segment.replaceAll("~", "~0").replaceAll("/", "~1"));
    } else if (typeof segment === "number" && Number.isInteger(segment)) {
      encoded.push("i:" + String(segment));
    } else {
      const got =
        typeof segment === "number" ? `non-integer number ${segment}` : jsTypeName(segment);
      throw new TypeError(`source site path segments must be text or integer values, got ${got}`);
    }
  }
  return "/" + encoded.join("/");
}

/**
 * Return one Compose origin rebased under a lexical `uses:` alias.
 *
 * The previous opaque key is encoded as one text segment; it is never parsed. The reserved
 * prefix can't overlap document-local keys, which always begin with `/`, so applying this
 * repeatedly composes namespaces without losing the boundary between aliases or nested keys.
 *
 * @throws TypeError when `alias` isn't text or `origin` isn't an `OriginRef`.
 * @throws RangeError when `alias` or the origin key is empty.
 */
export function scopeOrigin(alias: string, origin: OriginRef): OriginRef {
  if (typeof alias !== "string") throw new TypeError("origin scope alias must be text");
  if (alias === "") throw new RangeError("origin scope alias cannot be empty");
  if (!(origin instanceof OriginRef)) throw new TypeError("origin scope requires an OriginRef");
  const oldKey = new OriginRef(origin.key).key;
  return new OriginRef(SCOPED_ORIGIN_PREFIX + encodeSitePath(["uses", alias, "origin", oldKey]));
}

/**
 * Resolve one registered occurrence without copying or freezing its table: the concrete
 * frame, or null for an unknown site or one whose selected range is gone.
 *
 * @throws TypeError when `origin` isn't an `OriginRef`.
 * @throws SourceAccessError when a retained occurrence can't be replayed.
 */
function resolveRegisteredSite(
  sites: ReadonlyMap<string, SourceSite>,
  origin: OriginRef,
): DiagnosticFrame | null {
  if (!(origin instanceof OriginRef)) {
    throw new TypeError("source catalog resolve requires an OriginRef");
  }
  const site = sites.get(origin.key);
  if (site === undefined) return null;

  const cursor = site.document._resolveOccurrence(site.occurrencePath);
  const selector = site.selector;
  let range: SourceRange | null;
  if (selector === "value") {
    range = cursor.range;
  } else if (selector === "key") {
    range = cursor.keyRange;
  } else {
    const lineRanges = cursor.scalarSource?.constructedLineRanges ?? null;
    const index = selector.index;
    range =
      lineRanges !== null && 0 <= index && index < lineRanges.length ? lineRanges[index]! : null;
  }
  if (range === null) return null;
  return new DiagnosticFrame({ label: site.document.label, text: site.document.text, range });
}

/** Spell a site path as a tuple: `('nodes', 1)`, or `('a',)` for one segment. */
function tupleRepr(path: SourceSitePath): string {
  const items = path.map((segment) =>
    typeof segment === "string" ? reprText(segment) : String(segment),
  );
  return items.length === 1 ? `(${items[0]},)` : `(${items.join(", ")})`;
}

/** Return the error for registering `sitePath` twice, located at `range`. */
function duplicateSiteError(
  sitePath: SourceSitePath,
  range: SourceRange | null,
): SourceAccessError {
  return new SourceAccessError(`duplicate source catalog site ${tupleRepr(sitePath)}`, range);
}

/** Build a frozen site over frozen copies of `occurrencePath`, its steps, and a line selector. */
function siteOf(
  document: SourceDocument,
  occurrencePath: OccurrencePath,
  selector: SiteAnchor | ConstructedLine,
): SourceSite {
  return Object.freeze({
    document,
    occurrencePath: Object.freeze(occurrencePath.map((step) => Object.freeze({ ...step }))),
    selector: typeof selector === "object" ? Object.freeze({ index: selector.index }) : selector,
  });
}

/**
 * An immutable Compose resolver from opaque origins to diagnostic frames.
 *
 * The catalog owns no semantic program state. Its keys are local source identities, and its
 * values retain physical document occurrences. Unknown origins resolve to null so a caller
 * can compose resolvers later.
 */
export class SourceCatalog implements OriginResolver {
  readonly #sites: ReadonlyMap<string, SourceSite>;

  /**
   * Create a catalog over a defensive copy of `sites`, origin key to physical source site.
   *
   * @throws RangeError for a key that isn't non-empty text.
   * @throws TypeError for a value that isn't a source site.
   */
  constructor(sites: ReadonlyMap<string, SourceSite> = new Map()) {
    const copied = new Map<string, SourceSite>();
    for (const [key, site] of sites) {
      if (typeof key !== "string" || key === "") {
        throw new RangeError("source catalog keys must be non-empty strings");
      }
      const candidate: unknown = site;
      if (
        typeof candidate !== "object" ||
        candidate === null ||
        !((candidate as { document?: unknown }).document instanceof SourceDocument)
      ) {
        throw new TypeError("source catalog values must be source sites");
      }
      // A site from outside the builder may still be mutable; the snapshot keeps its own.
      copied.set(key, siteOf(site.document, site.occurrencePath, site.selector));
    }
    this.#sites = copied;
  }

  /**
   * Resolve one origin by replaying its retained physical occurrence: the concrete frame, or
   * null when `origin` is unknown or its retained anchor no longer has a range. A scoped child
   * identity must first route back to its local identity through a `ResolverTable`.
   *
   * @throws TypeError when `origin` isn't an `OriginRef`.
   * @throws SourceAccessError when the retained occurrence path can't be replayed.
   */
  resolve(origin: OriginRef): DiagnosticFrame | null {
    return resolveRegisteredSite(this.#sites, origin);
  }

  /**
   * Say whether one exact opaque origin belongs to this catalog snapshot. Membership reads
   * only the key table: it neither replays the occurrence nor resolves a frame. A non-origin
   * or malformed value is absent.
   */
  has(origin: unknown): boolean {
    return origin instanceof OriginRef && this.#sites.has(origin.key);
  }
}

/**
 * The mutable registration boundary that freezes immutable catalog snapshots.
 *
 * One builder serves one load of one document, so it also carries that load's tag ledger:
 * `consumedTags` holds the `TaggedNode.definitionId` of every tag a slot reader has read. The
 * loader compares it with `SourceDocument.taggedNodes()` to reject tags that type nothing.
 */
export class SourceCatalogBuilder {
  readonly #sites = new Map<string, SourceSite>();
  readonly consumedTags = new Set<number>();

  /**
   * Register one physical cursor under a deterministic semantic path, selecting its value's
   * range or its key's.
   *
   * A source-free cursor or a missing selected range returns null and doesn't reserve the
   * path. An already registered path is always an error, even when the later cursor is
   * unlocated or points at the same occurrence. Projected cursors keep their explicitly
   * supplied physical backing.
   *
   * @throws TypeError for an invalid path, cursor or anchor.
   * @throws RangeError for an empty path.
   * @throws SourceAccessError when `sitePath` is already registered.
   */
  register(
    sitePath: SourceSitePath,
    cursor: SourceCursor,
    anchor: SiteAnchor = "value",
  ): OriginRef | null {
    const key = encodeSitePath(sitePath);
    if (!(cursor instanceof SourceCursor)) {
      throw new TypeError("source catalog registration requires a SourceCursor");
    }
    if (anchor !== "value" && anchor !== "key") {
      throw new TypeError("source catalog anchor must be 'value' or 'key'");
    }
    const document = cursor.document;
    const range = anchor === "value" ? cursor.range : cursor.keyRange;
    if (this.#sites.has(key)) {
      throw duplicateSiteError(sitePath, range);
    }
    if (document === null || range === null) return null;

    this.#sites.set(key, siteOf(document, cursor.path, anchor));
    return new OriginRef(key);
  }

  /**
   * Register one exact constructed literal line by its zero-based index.
   *
   * Unchanged literal block scalars keep exact line ranges. Other scalar styles, virtual
   * cursors and value-changing projections return null without reserving the path, so the
   * caller can register a whole-value fallback. A site keeps the index, not a coordinate.
   *
   * @throws TypeError for an invalid path or cursor, or an index that isn't an integer.
   * @throws RangeError for an empty path.
   * @throws SourceAccessError for a negative or out-of-range index, or when `sitePath` is
   *   already registered.
   */
  registerConstructedLine(
    sitePath: SourceSitePath,
    cursor: SourceCursor,
    index: number,
  ): OriginRef | null {
    const key = encodeSitePath(sitePath);
    if (!(cursor instanceof SourceCursor)) {
      throw new TypeError("source catalog registration requires a SourceCursor");
    }
    if (!Number.isInteger(index)) {
      throw new TypeError("constructed line index must be an integer");
    }
    if (index < 0) {
      throw new SourceAccessError("constructed line index must be non-negative", cursor.range);
    }
    if (this.#sites.has(key)) {
      throw duplicateSiteError(sitePath, cursor.range);
    }

    const scalarSource = cursor.scalarSource;
    if (scalarSource === null || scalarSource.constructedLineRanges === null) return null;
    const lineRanges = scalarSource.constructedLineRanges;
    if (index >= lineRanges.length) {
      throw new SourceAccessError(
        `constructed line index ${index} is out of range`,
        scalarSource.range,
      );
    }

    const document = cursor.document;
    if (document === null) return null;
    this.#sites.set(key, siteOf(document, cursor.path, { index }));
    return new OriginRef(key);
  }

  /**
   * Resolve one currently registered site without freezing the builder.
   *
   * This read seam serves a load failure that must show a site before the file is finalized.
   * It replays the already-parsed document and never copies the mutable table.
   *
   * @throws TypeError when `origin` isn't an `OriginRef`.
   * @throws SourceAccessError when a retained occurrence can't be replayed.
   */
  resolveRegistered(origin: OriginRef): DiagnosticFrame | null {
    return resolveRegisteredSite(this.#sites, origin);
  }

  /** Return an immutable snapshot of all registrations so far. */
  freeze(): SourceCatalog {
    return new SourceCatalog(this.#sites);
  }
}
