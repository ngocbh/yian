/**
 * CODE: run deterministic TypeScript or JavaScript, either a file reference or an inline body.
 *
 * A `code` node carries one polymorphic `code:` field, classified once by shape:
 *
 * - a relative path to a `.ts`, `.js` or `.mjs` file plus `:fn` (`./helpers.ts:fn`,
 *   `../lib/x.mjs:run`) is **reference mode**: the named export, resolved against the flow's base
 *   directory, is called **in-process** with the node's inputs.
 * - anything else is **inline mode**: a body that reads the node's `inputs` and returns a value,
 *   run **in-process** with the same one-record calling convention, so a body promotes to a
 *   reference by moving it into a file.
 *
 * Both modes call with the node's bound typed input record (only its declared inputs, not the
 * pool) and return the node's one output value. Nothing is compiled or imported at construction;
 * the running comes with the body.
 *
 * Inline source is author-written and static (it lives in the flow file), and running the
 * operator's own code in-process is the capability reference mode already has (author is the
 * operator). That holds only while flows are operator-authored: CODE runs in-process with the
 * host's privileges and no isolation, so running a flow from an untrusted or agent-generated
 * source runs that source's code.
 *
 * Knows about:   nodes/base, nodes/contracts and diagnostics.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { OriginRef, reprText } from "../../diagnostics.ts";
import { Node, NodeKind } from "../base.ts";
import { mapOrigin, type OriginMapper, validateOptionalOrigin } from "../contracts.ts";

// A reference is a relative path to a TS or JS file, one colon and an export name. Anchored,
// so a body never matches.
const REFERENCE_RE = /^\.\.?\/[^\s:]+\.(?:ts|js|mjs):[A-Za-z_$][A-Za-z0-9_$]*$/;

/**
 * Classify a `code:` field value by shape into an execution mode.
 *
 * @returns `"reference"` for a relative `.ts`, `.js` or `.mjs` path plus `:fn`, else `"inline"`.
 */
export function classifyCodeSource(code: string): "reference" | "inline" {
  return REFERENCE_RE.test(code) ? "reference" : "inline";
}

/**
 * Select the finest source-neutral origin for one line of an inline body.
 *
 * @param line Candidate 1-based line of the authored body. Only an integer `number` from 1 to
 *   the number of line origins selects a line.
 * @param origins.codeOrigin The whole authored `code:` scalar's fallback.
 * @param origins.codeLineOrigins Exact authored line origins in constructed-value order.
 * @param origins.nodeOrigin The authored node's fallback when no CODE-specific origin exists.
 * @returns The exact line origin, then the whole-code origin, then the node origin.
 * @throws TypeError when an origin carrier has the wrong shape.
 */
export function selectInlineCodeOrigin(
  line: unknown,
  origins: {
    codeOrigin: OriginRef | null;
    codeLineOrigins: readonly OriginRef[];
    nodeOrigin: OriginRef | null;
  },
): OriginRef | null {
  const { codeOrigin, codeLineOrigins, nodeOrigin } = origins;
  validateOptionalOrigin(codeOrigin, "CODE source origin");
  validateOptionalOrigin(nodeOrigin, "node origin");
  validateLineOrigins(codeLineOrigins);
  if (typeof line === "number" && Number.isInteger(line) && line >= 1 && line <= codeLineOrigins.length) {
    return codeLineOrigins[line - 1]!;
  }
  return codeOrigin ?? nodeOrigin;
}

/** Require the exact-line origins to be an array of `OriginRef`s. */
function validateLineOrigins(codeLineOrigins: unknown): void {
  if (!Array.isArray(codeLineOrigins)) throw new TypeError("CODE line origins must be an array");
  for (const origin of codeLineOrigins) {
    if (!(origin instanceof OriginRef)) throw new TypeError("CODE line origins must contain only OriginRef values");
  }
}

/** Require source-neutral CODE diagnostic carriers. */
function validateCodeOrigins(codeOrigin: unknown, codeLineOrigins: unknown): void {
  validateOptionalOrigin(codeOrigin, "CODE source origin");
  validateLineOrigins(codeLineOrigins);
}

/**
 * Run deterministic code: a file reference or an inline body, both in-process.
 *
 * The one `code:` field is classified by shape at construction with `classifyCodeSource`, and the
 * mode is stored. The node never touches the pool: its body returns `new Output(value)`.
 */
export class CodeNode extends Node {
  readonly kind = NodeKind.CODE;
  /** The `code:` field: a `./file.ts:fn` reference or an inline body. */
  code: string;
  /** The directory a reference resolves against; `null` when not known. Not fingerprinted. */
  baseDir: string | null;
  /** The authored identity of the whole `code:` scalar. Diagnostic-only. */
  codeOrigin: OriginRef | null;
  /** The authored identities of the exact constructed body lines, in order. Diagnostic-only. */
  codeLineOrigins: readonly OriginRef[];
  /** The execution mode `classifyCodeSource` gives `code`. */
  mode: "reference" | "inline";

  /**
   * Build a CODE node. Nothing is compiled or imported here.
   *
   * @param nodeId The node's unique id.
   * @param options.code The `code:` field: a `./file.ts:fn` reference or an inline body.
   * @param options.baseDir The directory a reference resolves against.
   * @param options.codeOrigin The authored identity of the whole `code:` scalar.
   * @param options.codeLineOrigins The authored identities of the exact constructed body lines in
   *   order; empty when line-level provenance is unavailable. The node keeps a frozen copy.
   * @param options.title A display title.
   * @throws TypeError when `codeOrigin` isn't an `OriginRef` or `null`, or `codeLineOrigins` isn't
   *   an array of `OriginRef`s.
   */
  constructor(
    nodeId: string,
    options: {
      code: string;
      baseDir?: string | null;
      codeOrigin?: OriginRef | null;
      codeLineOrigins?: readonly OriginRef[];
      title?: string | null;
    },
  ) {
    const { code, baseDir = null, codeOrigin = null, codeLineOrigins = [], title = null } = options;
    super(nodeId, { title });
    validateCodeOrigins(codeOrigin, codeLineOrigins);
    this.code = code;
    this.baseDir = baseDir;
    this.codeOrigin = codeOrigin;
    this.codeLineOrigins = Object.freeze([...codeLineOrigins]);
    this.mode = classifyCodeSource(code);
  }

  /** Yield the common origins, then the whole-code origin and the exact line origins. */
  override *iterLocalOrigins(seen: Set<object> = new Set()): Generator<OriginRef> {
    validateCodeOrigins(this.codeOrigin, this.codeLineOrigins);
    yield* super.iterLocalOrigins(seen);
    if (this.codeOrigin !== null) yield this.codeOrigin;
    yield* this.codeLineOrigins;
  }

  /**
   * Rewrite the common and the CODE-specific origins on a copied CODE node.
   *
   * Assigns `codeOrigin` and a fresh frozen `codeLineOrigins` array.
   */
  override rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    validateCodeOrigins(this.codeOrigin, this.codeLineOrigins);
    super.rewriteLocalOrigins(mapper, seen);
    this.codeOrigin = mapOrigin(this.codeOrigin, mapper);
    this.codeLineOrigins = Object.freeze(this.codeLineOrigins.map((origin) => mapOrigin(origin, mapper)!));
  }

  /** The reference and inline bodies; not ported yet. */
  override forward(): never {
    throw new Error(`node ${reprText(this.id)} (CodeNode) forward is not ported yet (T4-9)`);
  }
}
