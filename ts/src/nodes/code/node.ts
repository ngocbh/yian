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
 * return the node's one output value. Nothing is compiled or imported at construction.
 *
 * An inline body is stripped of its types and built as a strict async function of `inputs` each
 * time the node runs. `inputs` is a frozen host copy of the record (ints as `number`s, big ones as
 * `bigint`s) behind a proxy that throws on an undeclared name. What the body throws rejects as a
 * `CodeError` reading `<name>: <message>`, located at the authored line of its first body frame
 * when it has one; a body returning `undefined` fails; its result goes through host intake, and a
 * value intake refuses fails `contract.output`. The built-in identity returns its field as bound.
 *
 * Inline source is author-written and static (it lives in the version-controlled flow file), so
 * there is no injection vector into the code itself, and running it in-process is the same
 * capability reference mode already has. Loading a flow from an untrusted or agent-generated
 * source runs that source's code when the flow runs.
 *
 * Knows about:   nodes/base, nodes/contracts, diagnostics, typesys, platform and `node:*`.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { type FailureCode, OriginRef, reprText } from "../../diagnostics.ts";
import { stripTypes } from "../../platform.ts";
import {
  fromHost,
  isRecord,
  jsTypeName,
  toHost,
  TypeCheckError,
  type Value,
  writeAsText,
} from "../../typesys/values.ts";
import { markerMessage, Node, NodeKind, Output } from "../base.ts";
import { mapOrigin, type OriginMapper, outputContractOrigin, validateOptionalOrigin } from "../contracts.ts";

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
    const got = code === null ? "null" : jsTypeName(code);
    throw new TypeError(`CODE code must be text or a built-in operation, got ${got}`);
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

/**
 * Require source-neutral CODE diagnostic carriers; only the selector passes a node origin.
 *
 * @throws TypeError when `codeOrigin` or `nodeOrigin` isn't an `OriginRef` or `null`, or
 *   `codeLineOrigins` isn't an array of `OriginRef`s.
 */
export function validateCodeOrigins(codeOrigin: unknown, codeLineOrigins: unknown, nodeOrigin: unknown = null): void {
  validateOptionalOrigin(codeOrigin, "CODE source origin");
  validateOptionalOrigin(nodeOrigin, "node origin");
  if (!Array.isArray(codeLineOrigins)) throw new TypeError("CODE line origins must be an array");
  for (const origin of codeLineOrigins) {
    if (!(origin instanceof OriginRef)) throw new TypeError("CODE line origins must contain only OriginRef values");
  }
}

/** An inline body: an async function of the `inputs` record. */
type CodeBody = (inputs: Readonly<Record<string, unknown>>) => Promise<unknown>;

// The constructor isn't a global; every async function's `constructor` is it.
const AsyncFunction = (async () => {}).constructor as new (...args: string[]) => CodeBody;

/** The number of lines the runtime's function constructor puts before a body; measured once. */
let headerLines: number | null = null;

/** Return how many lines the runtime's function constructor puts before a body. */
function headerLineCount(): number {
  if (headerLines === null) {
    const text = new AsyncFunction("inputs", "/*yian*/").toString();
    headerLines = text.slice(0, text.indexOf("/*yian*/")).split("\n").length - 1;
  }
  return headerLines;
}

/**
 * Strip an inline body's types and build it as a strict async function of `inputs`.
 *
 * The `sourceURL` comment names the body's stack frames `yian-code:<nodeId>:<line>:<column>`, and
 * the strict prologue shares the first line, so frame lines stay body lines plus the header.
 *
 * @throws TypeStripError when the body doesn't parse or strip.
 * @throws SyntaxError when the constructor refuses the stripped body.
 */
async function compileBody(nodeId: string, body: string): Promise<CodeBody> {
  const js = await stripTypes(body);
  return new AsyncFunction("inputs", '"use strict"; ' + js + "\n//# sourceURL=yian-code:" + nodeId);
}

/** Return the 1-based body line of the first `yian-code:<nodeId>` frame in `error`'s stack, else `null`. */
function bodyLine(error: unknown, nodeId: string): number | null {
  const stack = error instanceof Error ? error.stack : undefined;
  if (typeof stack !== "string") return null;
  const marker = `yian-code:${nodeId}:`;
  for (const line of stack.split("\n")) {
    const at = line.indexOf(marker);
    if (at === -1) continue;
    const match = /^\d+/.exec(line.slice(at + marker.length));
    if (match !== null) return Number(match[0]) - headerLineCount();
  }
  return null;
}

/** Return the error a CODE body throws when it reads an input the node doesn't declare. */
function undeclaredInput(name: string): ReferenceError {
  return new ReferenceError(`CODE input ${reprText(name)} is not declared`);
}

/**
 * Return `record` behind a proxy that throws on a string key the record doesn't hold.
 *
 * The keys the runtime itself probes pass: inherited `Object.prototype` members, `then` (promise
 * resolution), `toJSON` (`JSON.stringify`) and symbols.
 */
function inputsProxy(record: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  return new Proxy(record, {
    get(target, key, receiver) {
      if (typeof key === "symbol" || key in target || key === "then" || key === "toJSON") {
        return Reflect.get(target, key, receiver);
      }
      throw undeclaredInput(key);
    },
  });
}

/**
 * A value CODE's own code threw, wrapped so the failure reads `<name>: <message>`.
 *
 * A non-`Error` reads as its text form when it is a yian value, else as `String(value)`. The
 * stack is the thrown error's, so a failure's traceback shows the code's own frames.
 */
export class CodeError extends Error {
  override name = "CodeError";
  /** The exact value the code threw. */
  readonly original: unknown;

  /** Wrap `original`, also kept as the `cause`. */
  constructor(original: unknown) {
    super(codeErrorMessage(original), { cause: original });
    this.original = original;
    if (original instanceof Error) this.stack = original.stack;
  }
}

/** Return the text of a value CODE's code threw. */
function codeErrorMessage(original: unknown): string {
  if (original instanceof Error) return `${original.name}: ${original.message}`;
  try {
    return writeAsText(fromHost(original, "error"));
  } catch {
    return markerMessage(original);
  }
}

/**
 * Run deterministic code: a file reference or an inline body, both in-process, or a built-in.
 *
 * The one `code:` field is classified by shape at construction, and the mode is stored: text with
 * `classifyCodeSource`, a built-in form as `"builtin"`. A tagged value entry lowers to the built-in
 * identity form. The node never touches the pool: its body returns `new Output(value)`. An inline
 * body runs in-process, built afresh on each run, and a built-in reads the bound record directly.
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

  /**
   * Return the failure code for one body error: `contract.output` for a result host intake
   * refuses, else the base's.
   */
  override exceptionCode(error: unknown): FailureCode {
    return error instanceof TypeCheckError ? "contract.output" : super.exceptionCode(error);
  }

  /**
   * Return the most precise authored origin for one body error.
   *
   * A built-in's is the node's. A result host intake refuses is located in the output contract.
   * Otherwise the first `yian-code:<id>` frame of what the code threw selects the authored line,
   * falling back to the whole `code:` origin, then the node's (a thrown non-`Error` has no stack,
   * and a native error thrown after an `await` has no body frame on Bun).
   */
  override exceptionOrigin(error: unknown): OriginRef | null {
    if (this.mode === "builtin") return super.exceptionOrigin(error);
    if (error instanceof TypeCheckError) return outputContractOrigin(this.outputContract, error.path);
    const line = error instanceof CodeError ? bodyLine(error.original, this.id) : null;
    return selectInlineCodeOrigin(line, {
      codeOrigin: this.codeOrigin,
      codeLineOrigins: this.codeLineOrigins,
      nodeOrigin: this.origin,
    });
  }

  /**
   * Run the code on the bound input record and return its one value.
   *
   * The built-in identity returns the record's field as bound. An inline body gets `inputs`, a
   * frozen host copy (ints as `number`s, big ones as `bigint`s) behind a proxy that throws on an
   * undeclared name; what it throws rejects as a `CodeError`, and its result goes through host
   * intake (a top-level `return inputs` returns the record).
   *
   * @throws ReferenceError when the built-in's field isn't in the record.
   * @throws Error when a body returns `undefined`.
   * @throws TypeCheckError when host intake refuses the result.
   */
  override async forward(inputs: Record<string, unknown>): Promise<Output> {
    const code = this.code;
    if (typeof code !== "string") {
      if (!Object.hasOwn(inputs, code.field)) throw undeclaredInput(code.field);
      return new Output(inputs[code.field]);
    }
    if (this.mode === "reference") {
      throw new Error(`node ${reprText(this.id)} (CodeNode) forward is not ported yet (T4-9)`);
    }
    const body = await compileBody(this.id, code);
    const record = Object.freeze(toHost(inputs as Value) as Record<string, unknown>);
    const proxy = inputsProxy(record);
    let result: unknown;
    try {
      result = await body(proxy);
    } catch (error) {
      throw new CodeError(error);
    }
    if (result === undefined) throw new Error("CODE body returned no value; return null for no value");
    return new Output(fromHost(result === proxy ? record : result, "output"));
  }
}
