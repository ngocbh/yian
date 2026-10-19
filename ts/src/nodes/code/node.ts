/**
 * CODE: run deterministic TypeScript or JavaScript (a file reference or an inline body), or a
 * built-in operation.
 *
 * A `code` node carries one polymorphic `code:` field, classified once by shape:
 *
 * - a relative path to a `.ts`, `.js` or `.mjs` file plus `:fn` (`./helpers.ts:fn`,
 *   `../lib/x.mjs:run`) is **reference mode**: the named export, resolved against the flow's base
 *   directory, is called **in-process** with the node's inputs.
 * - any other text is **inline mode**: a body that reads the node's `inputs` and returns a value,
 *   run **in-process** with the same one-record calling convention, so a body promotes to a
 *   reference by moving it into a file.
 * - a `{builtin: "identity", field: <input name>}` object is **built-in mode**: the node returns
 *   that input unchanged, with no module to load. A tagged value entry in `nodes:` lowers to this
 *   form rather than to code text.
 *
 * All modes read the node's bound typed input record (only its declared inputs, not the pool) and
 * return the node's one output value.
 *
 * Only the node's shell is here. Nothing is compiled or imported at construction; the body
 * (importing a reference, running an inline body or a built-in, and mapping a failure to its
 * line) is not ported yet and comes with `forward`.
 *
 * Inline source is author-written and static (it lives in the version-controlled flow file), so
 * there is no injection vector into the code itself, and running it in-process is the same
 * capability reference mode already has. Loading a flow from an untrusted or agent-generated
 * source runs that source's code when the flow runs.
 *
 * Knows about:   nodes/base, nodes/contracts, diagnostics and typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { OriginRef, reprText } from "../../diagnostics.ts";
import { isRecord, jsTypeName } from "../../typesys/values.ts";
import { Node, NodeKind } from "../base.ts";
import { mapOrigin, type OriginMapper, validateOptionalOrigin } from "../contracts.ts";

// A reference is a relative path to a TS or JS file, one colon and an ASCII identifier export
// name. Anchoring plus a whitespace-free path keep multi-token bodies out.
const REFERENCE_RE = /^\.\.?\/[^\s:]+\.(?:ts|js|mjs):[A-Za-z_$][A-Za-z0-9_$]*$/;

/** The built-in identity operation: return the input named `field` unchanged. */
export interface BuiltinCode {
  readonly builtin: "identity";
  readonly field: string;
}

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
  validateCodeOrigins(codeOrigin, codeLineOrigins, nodeOrigin);
  if (typeof line === "number" && Number.isInteger(line) && line >= 1 && line <= codeLineOrigins.length) {
    return codeLineOrigins[line - 1]!;
  }
  return codeOrigin ?? nodeOrigin;
}

/**
 * Require a `code:` value to be text or exactly `{builtin: "identity", field: <input name>}`.
 *
 * `field` names the input the operation returns (compose sets it to the value entry's id); it
 * isn't checked against the declared inputs here.
 *
 * @returns The text itself, or a frozen copy of the built-in form.
 */
function validateCode(code: unknown): string | BuiltinCode {
  if (typeof code === "string") return code;
  if (!isRecord(code)) {
    throw new TypeError(`CODE code must be text or a built-in operation, got ${jsTypeName(code)}`);
  }
  const keys = Object.keys(code);
  if (
    keys.length !== 2 ||
    !keys.includes("builtin") ||
    !keys.includes("field") ||
    code.builtin !== "identity" ||
    typeof code.field !== "string"
  ) {
    throw new TypeError('CODE built-in must be {builtin: "identity", field: <input name>}');
  }
  if (code.field === "") throw new RangeError("CODE built-in field must be non-empty");
  return Object.freeze({ builtin: "identity", field: code.field });
}

/** Require source-neutral CODE diagnostic carriers; only the selector passes a node origin. */
function validateCodeOrigins(codeOrigin: unknown, codeLineOrigins: unknown, nodeOrigin: unknown = null): void {
  validateOptionalOrigin(codeOrigin, "CODE source origin");
  validateOptionalOrigin(nodeOrigin, "node origin");
  if (!Array.isArray(codeLineOrigins)) throw new TypeError("CODE line origins must be an array");
  for (const origin of codeLineOrigins) {
    if (!(origin instanceof OriginRef)) throw new TypeError("CODE line origins must contain only OriginRef values");
  }
}

/**
 * Run deterministic code: a file reference or an inline body, both in-process, or a built-in.
 *
 * The one `code:` field is classified by shape at construction, and the mode is stored: text with
 * `classifyCodeSource`, a built-in form as `"builtin"`. A tagged value entry lowers to the built-in
 * identity form. The node never touches the pool: its body returns `new Output(value)`. Running
 * any mode, the built-in included, is not ported yet and comes with `forward`.
 */
export class CodeNode extends Node {
  readonly kind = NodeKind.CODE;
  /**
   * The `code:` field: a `./file.ts:fn` reference, an inline body, or a frozen built-in form (what
   * a value entry lowers to).
   */
  code: string | BuiltinCode;
  /** The directory a reference resolves against; `null` when not known. Not fingerprinted. */
  baseDir: string | null;
  /** The authored identity of the whole `code:` scalar. Diagnostic-only. */
  codeOrigin: OriginRef | null;
  /** The authored identities of the exact constructed body lines, in order. Diagnostic-only. */
  codeLineOrigins: readonly OriginRef[];
  /** The execution mode: `classifyCodeSource`'s for text, `"builtin"` for a built-in form. */
  mode: "reference" | "inline" | "builtin";

  /**
   * Build a CODE node. Nothing is compiled or imported here.
   *
   * @param nodeId The node's unique id.
   * @param options.code The `code:` field: a `./file.ts:fn` reference, an inline body, or a
   *   `{builtin: "identity", field}` form. The node keeps a frozen copy of a built-in form.
   * @param options.baseDir The directory a reference resolves against.
   * @param options.codeOrigin The authored identity of the whole `code:` scalar.
   * @param options.codeLineOrigins The authored identities of the exact constructed body lines in
   *   order; empty when line-level provenance is unavailable. The node keeps a frozen copy.
   * @param options.title A display title.
   * @throws TypeError when `code` is neither text nor exactly `{builtin: "identity", field: <text>}`,
   *   `codeOrigin` isn't an `OriginRef` or `null`, or `codeLineOrigins` isn't an array of
   *   `OriginRef`s.
   * @throws RangeError when a built-in form's `field` is empty.
   */
  constructor(
    nodeId: string,
    options: {
      code: string | BuiltinCode;
      baseDir?: string | null;
      codeOrigin?: OriginRef | null;
      codeLineOrigins?: readonly OriginRef[];
      title?: string | null;
    },
  ) {
    const { code, baseDir = null, codeOrigin = null, codeLineOrigins = [], title = null } = options;
    super(nodeId, { title });
    this.code = validateCode(code);
    validateCodeOrigins(codeOrigin, codeLineOrigins);
    this.baseDir = baseDir;
    this.codeOrigin = codeOrigin;
    this.codeLineOrigins = Object.freeze([...codeLineOrigins]);
    this.mode = typeof this.code === "string" ? classifyCodeSource(this.code) : "builtin";
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

  /** The reference, inline and built-in bodies; not ported yet. */
  override forward(): never {
    throw new Error(`node ${reprText(this.id)} (CodeNode) forward is not ported yet (T4-9)`);
  }
}
