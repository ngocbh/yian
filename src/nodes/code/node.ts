/**
 * CODE: run deterministic TypeScript or JavaScript (a file reference or an inline body), or a
 * built-in operation.
 *
 * A `code` node carries one polymorphic `code:` field, classified once by shape:
 *
 * - a relative path to a `.ts`, `.js` or `.mjs` file plus `:fn` (`./helpers.ts:fn`,
 *   `../lib/x.mjs:run`) is **reference mode**: the named export, resolved against the flow's base
 *   directory, is called **in a worker thread** with the node's inputs.
 * - any other text is **inline mode**: a body that reads the node's `inputs` and returns a value,
 *   run **in a worker thread** with the same one-record calling convention, so a body promotes to
 *   a reference by moving it into a file.
 * - a `{builtin: "identity", field: <input name>}` object is **built-in mode**: the node returns
 *   that input unchanged, with no module to load. A tagged value entry in `nodes:` lowers to this
 *   form rather than to code text.
 *
 * All modes read the node's bound typed input record (only its declared inputs, not the pool) and
 * return the node's one output value. Nothing is compiled or imported at construction; the
 * awaited load step's `check()` builds an inline body without calling it, refuses a Python-style
 * `module:function` value, and requires a reference's file to exist (and, when the load has a code
 * root, to resolve inside it); it runs no code.
 *
 * A reference or an inline body runs on a worker thread (`nodes/code/worker`), one call at a time
 * per thread. Idle workers are reused (at most `availableParallelism()` wait), and an idle worker
 * never keeps the process alive; module state and globals may persist between calls that land on
 * the same thread, but nothing guarantees it. A referenced module is imported once per thread and
 * cached there, so an edited file takes effect in a new process or a new thread. An inline body is
 * stripped of its types and built as a strict async function of `inputs`. Both are called the
 * same way: `inputs` is a frozen host copy of the record (ints as `number`s, big ones as
 * `bigint`s, files as `FileRef`s) behind a proxy that throws on an undeclared name; the reference's
 * named export must be a function. What the code throws (a reference's module while it loads
 * included, and a late timer throw or unhandled rejection while it runs) rejects as a `CodeError`
 * reading `<name>: <message>`; an inline body's is located at the authored line of its first body
 * frame when it has one, a reference's at the whole `code:` value. Its stack names an inline
 * body's frames after the node's `sourceFile` (the flow's file label) and the body's own line. A
 * result of `undefined` fails; a result goes through host intake in the worker, and a value intake
 * refuses fails `contract.output`. A worker that exits before the body returns rejects with its
 * exit code: a `process.exit`, or a body awaiting a promise nothing can resolve when no earlier
 * body on its thread left work running (a `setInterval`, say); otherwise that body waits until it
 * is stopped.
 * A body shouldn't leave work running after it returns: a late error from it fails the call its
 * thread runs then, or is printed as a console error line when the thread is idle, and the thread
 * is retired. The worker forwards the code's `console` calls to this thread's `console`. The
 * built-in identity returns its field as bound, with no worker.
 *
 * Inline source is author-written and static (it lives in the version-controlled flow file), so
 * there is no injection vector into the code itself, and running it is the same capability
 * reference mode already has. A worker thread is not a sandbox: the code shares this process's
 * privileges, file system and live environment variables (a body's change reaches this thread,
 * and this thread's later change reaches the body). Loading a flow from an untrusted or
 * agent-generated source runs that source's code when the flow runs.
 *
 * Knows about:   nodes/base, nodes/contracts, nodes/code/worker, diagnostics, typesys, defaults,
 *                platform and `node:*`.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { realpathSync, statSync } from "node:fs";
import { availableParallelism } from "node:os";
import { isAbsolute, relative, resolve, sep } from "node:path";
import { SHARE_ENV, Worker } from "node:worker_threads";

import { current } from "../../defaults.ts";
import { OriginRef, reprText } from "../../diagnostics.ts";
import { TypeStripError } from "../../platform.ts";
import {
  fromHost,
  isRecord,
  jsTypeName,
  toHost,
  TypeCheckError,
  type Value,
  writeAsText,
} from "../../typesys/values.ts";
import { HostOutputError, hostOutput, markerMessage, Node, type NodeCaps, NodeKind, Output } from "../base.ts";
import { mapOrigin, type OriginMapper, validateOptionalOrigin } from "../contracts.ts";
import {
  AsyncFunction,
  type CodeJob,
  type CodeMessage,
  type CodeReply,
  compileBody,
  packFiles,
  splitReference,
  STRICT_PROLOGUE,
  type ThrownError,
  undeclaredInput,
  unpackFiles,
} from "./worker.ts";

// A reference is a relative path to a TS or JS file, one colon and an ASCII identifier export
// name. Anchoring plus a whitespace-free path keep multi-token bodies out.
const REFERENCE_RE = /^\.\.?\/[^\s:]+\.(?:ts|js|mjs):[A-Za-z_$][A-Za-z0-9_$]*$/;

// A module reference (`pkg.mod:fn`), which yian doesn't run. Read as TS it is a labeled
// statement, so the load check refuses it rather than run it as a body.
const MODULE_REFERENCE_RE = /^[A-Za-z_][\w.]*:[A-Za-z_]\w*$/;

// Mirrors the CLI's `leavesRoot` in `cli/run.ts`; nodes may not import the CLI.
/**
 * Whether a path relative to a root (`relative(root, file)`) leaves that root: `..`, a `../`
 * prefix, or an absolute path (another drive).
 */
function leavesRoot(rel: string): boolean {
  return rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel);
}

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

/**
 * Return `stack` with each `yian-code:<nodeId>:<line>:<column>` location renamed
 * `<sourceFile>:<body line>:<body column>`: the line less the constructor's header, and a
 * first-line column less the strict prologue sharing that line. A message holding that text is
 * renamed too when its numbers read as a body location.
 */
function labelBodyFrames(stack: string, nodeId: string, sourceFile: string): string {
  const marker = `yian-code:${nodeId}:`;
  const [head, ...rest] = stack.split(marker);
  return rest.reduce((text, part) => {
    const match = /^(\d+):(\d+)/.exec(part);
    if (match === null) return text + marker + part;
    const line = Number(match[1]) - headerLineCount();
    const column = Number(match[2]) - (line === 1 ? STRICT_PROLOGUE.length : 0);
    // Not a body location (a message holding the marker text): leave it.
    if (line < 1 || column < 1) return text + marker + part;
    return `${text}${sourceFile}:${line}:${column}${part.slice(match[0].length)}`;
  }, head!);
}

/**
 * A value CODE's own code threw, wrapped so the failure reads `<name>: <message>`.
 *
 * A non-`Error` reads as its text form when it is a yian value, else as `String(value)`. The
 * stack is the thrown error's, so a failure's traceback shows the code's own frames (`forward`
 * renames an inline body's after the node's `sourceFile`).
 */
export class CodeError extends Error {
  override name = "CodeError";
  /**
   * A copy of what the code threw, rebuilt from the worker's reply: an `Error` (`rebuildError`), a
   * yian value, or the thrown value's text.
   */
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

// `bun build --compile` bundles the worker at `nodes/code/worker.js` beside the bundle's entry,
// whose URL this module then has.
const WORKER_URL = import.meta.url.endsWith(".ts")
  ? new URL("./worker.ts", import.meta.url)
  : import.meta.url.endsWith("/node.js")
    ? new URL("./worker.js", import.meta.url)
    : new URL("./nodes/code/worker.js", import.meta.url);

/** The standard error classes a worker's error is rebuilt as, by name. */
const ERROR_CLASSES: Readonly<Record<string, ErrorConstructor>> = {
  Error,
  EvalError,
  RangeError,
  ReferenceError,
  SyntaxError,
  TypeError,
  URIError,
};

/**
 * Rebuild an error a worker sent: a `TypeStripError` with its line and column, a standard class
 * by name, else an `Error` carrying the name. The worker's stack is kept, so its body frames
 * still map to authored lines.
 */
function rebuildError(fields: ThrownError): Error {
  let error: Error;
  if (fields.name === "TypeStripError" && fields.line !== undefined && fields.column !== undefined) {
    error = new TypeStripError(fields.message, fields.line, fields.column);
  } else if (Object.hasOwn(ERROR_CLASSES, fields.name)) {
    error = new ERROR_CLASSES[fields.name]!(fields.message);
  } else {
    error = new Error(fields.message);
    error.name = fields.name;
  }
  if (fields.stack !== undefined) error.stack = fields.stack;
  return error;
}

/** Return the error a failed reply rejects with. */
function replyError(reply: Exclude<CodeReply, { ok: true }>): Error {
  if (reply.as === "output") {
    const cause = new TypeCheckError(reply.message, { path: reply.path });
    return new HostOutputError(reply.message, { path: reply.path, cause });
  }
  const { thrown } = reply;
  const original =
    thrown.kind === "error" ? rebuildError(thrown) : thrown.kind === "value" ? unpackFiles(thrown.value) : thrown.text;
  return reply.as === "plain" && original instanceof Error ? original : new CodeError(original);
}

/** A call running on a worker: its job id and how it settles. */
interface Call {
  readonly id: number;
  readonly resolve: (reply: CodeReply) => void;
  readonly reject: (error: unknown) => void;
}

/** A started worker and the call it runs; `call` is `null` while it is idle or out of service. */
interface CodeWorker {
  readonly worker: Worker;
  call: Call | null;
}

/** At most this many idle workers wait for a call. */
const POOL_SIZE = availableParallelism();

/** The idle workers, the most recently used last. */
const idle: CodeWorker[] = [];

/** The last job id handed out. */
let lastJobId = 0;

/** Take `entry` out of the idle pool, if it is there. */
function leavePool(entry: CodeWorker): void {
  const at = idle.indexOf(entry);
  if (at !== -1) idle.splice(at, 1);
}

/** Take `entry` out of service: out of the pool, its call (returned) detached, the thread stopped. */
function retire(entry: CodeWorker): Call | null {
  const call = entry.call;
  entry.call = null;
  leavePool(entry);
  void entry.worker.terminate();
  return call;
}

/** Start a worker whose listeners route to its current call for its whole life. */
function startWorker(): CodeWorker {
  // `SHARE_ENV`: a body reads and writes the host's live environment, as it did in-process.
  const worker = new Worker(WORKER_URL, { env: SHARE_ENV, workerData: { yianCodeWorker: true } });
  const entry: CodeWorker = { worker, call: null };
  const failed = (error: unknown): void => {
    retire(entry)?.reject(new Error(`CODE worker failed: ${markerMessage(error)}`, { cause: error }));
  };
  worker.on("message", (message: CodeMessage) => {
    if ("console" in message) {
      console[message.console](message.text);
      // A late error leaves the thread suspect (`bun test` kills it); an idle one stops now.
      if (message.late === true && entry.call === null) retire(entry);
      return;
    }
    const call = entry.call;
    if (call === null || message.id !== call.id) return;
    entry.call = null;
    if (message.late === true || idle.length >= POOL_SIZE) {
      void worker.terminate();
    } else {
      // An idle worker never keeps this process alive.
      worker.unref();
      idle.push(entry);
    }
    call.resolve(message);
  });
  worker.on("exit", (code: number) => {
    leavePool(entry);
    const call = entry.call;
    entry.call = null;
    call?.reject(new Error(`CODE worker exited with code ${code} before the body returned`));
  });
  worker.on("error", failed);
  worker.on("messageerror", failed);
  return entry;
}

/**
 * Run `job` on an idle worker, or a new one, and return its reply.
 *
 * @throws The signal's reason when `signal` aborts first; the worker is terminated before this
 *   rejects.
 * @throws Error when the worker exits before it replies, or fails to start.
 * @throws The clone's error when `job` can't be sent; the worker is retired.
 */
function runInWorker(job: CodeJob, signal: AbortSignal | undefined): Promise<CodeReply> {
  return new Promise((resolve, reject) => {
    const entry = idle.pop() ?? startWorker();
    const call: Call = {
      id: ++lastJobId,
      resolve: (reply) => {
        signal?.removeEventListener("abort", onAbort);
        resolve(reply);
      },
      reject: (error) => {
        signal?.removeEventListener("abort", onAbort);
        reject(error);
      },
    };
    const onAbort = (): void => {
      if (entry.call !== call) return;
      entry.call = null;
      const settle = () => call.reject(signal!.reason);
      entry.worker.terminate().then(settle, settle);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    entry.call = call;
    entry.worker.ref();
    try {
      entry.worker.postMessage({ ...job, id: call.id });
    } catch (error) {
      // A job the clone refuses (a function input from a direct caller) never reaches the worker.
      retire(entry)?.reject(error);
    }
  });
}

/**
 * Run deterministic code: a file reference or an inline body, both in a worker thread, or a built-in.
 *
 * The one `code:` field is classified by shape at construction, and the mode is stored: text with
 * `classifyCodeSource`, a built-in form as `"builtin"`. A tagged value entry lowers to the built-in
 * identity form. The node never touches the pool: its body returns `new Output(value)`. A
 * reference's function and an inline body run on a worker thread (an idle one reused), with one
 * calling convention, and a built-in reads the bound record directly.
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
  /** The flow's file label an inline body's stack frames are named after. Not fingerprinted. */
  sourceFile: string;
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
   * @param options.sourceFile The flow's file label a failing inline body's stack frames are named
   *   after (default `"<inline-code>"`).
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
      sourceFile?: string;
      codeOrigin?: OriginRef | null;
      codeLineOrigins?: readonly OriginRef[];
      title?: string | null;
    },
  ) {
    const {
      code,
      baseDir = null,
      sourceFile = "<inline-code>",
      codeOrigin = null,
      codeLineOrigins = [],
      title = null,
    } = options;
    super(nodeId, { title });
    this.code = validateCode(code);
    validateCodeOrigins(codeOrigin, codeLineOrigins);
    this.baseDir = baseDir;
    this.sourceFile = sourceFile;
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
   * Return the most precise authored origin for one body error.
   *
   * A built-in's is the node's, and a result host intake refuses is located by the base, in the
   * output contract. A reference's is the whole `code:` origin, else the node's. Otherwise the first `yian-code:<id>`
   * frame of what the code threw selects the authored line, falling back to the whole `code:`
   * origin, then the node's (a thrown non-`Error` has no stack, and a native error thrown after an
   * `await` has no body frame on Bun).
   */
  override exceptionOrigin(error: unknown): OriginRef | null {
    if (this.mode === "builtin" || error instanceof HostOutputError) return super.exceptionOrigin(error);
    if (this.mode === "reference") return this.codeOrigin ?? super.exceptionOrigin(error);
    const line = error instanceof CodeError ? bodyLine(error.original, this.id) : null;
    return selectInlineCodeOrigin(line, {
      codeOrigin: this.codeOrigin,
      codeLineOrigins: this.codeLineOrigins,
      nodeOrigin: this.origin,
    });
  }

  /**
   * Check the code without running any of it: the load step's node half.
   *
   * A built-in passes. A reference's file, resolved against `baseDir` (the current directory
   * when `null`), must be a regular file; its module isn't imported, so a missing export is
   * found when the node runs. An inline body that has Python's `module:function` shape is
   * refused; otherwise it is stripped of its types and built, never called.
   *
   * @param options.codeRoot The real path of the directory a reference's file must resolve
   *   inside, symlinks followed (a file names only itself); `null` (the default) confines nothing.
   * @throws RangeError when a reference's file isn't a regular file, or resolves outside
   *   `options.codeRoot`, or an inline body looks like a Python reference.
   * @throws TypeStripError when the body doesn't parse or strip.
   * @throws SyntaxError when the function constructor refuses the stripped body.
   */
  async check(options: { codeRoot?: string | null } = {}): Promise<void> {
    const code = this.code;
    if (typeof code !== "string") return;
    if (this.mode === "reference") {
      const { path } = splitReference(code);
      const dir = this.baseDir ?? process.cwd();
      const file = resolve(dir, path);
      if (statSync(file, { throwIfNoEntry: false })?.isFile() !== true) {
        throw new RangeError(`\`code:\` file ${reprText(path)} does not exist in ${reprText(dir)}`);
      }
      const { codeRoot = null } = options;
      if (codeRoot !== null) {
        // The native realpath, like the loader's `fsp.realpath` for the root, so both give the
        // on-disk case and compare equal on a case-insensitive disk, on Bun and Node alike.
        const real = realpathSync.native(file);
        if (leavesRoot(relative(codeRoot, real))) {
          throw new RangeError(
            `\`code:\` file ${reprText(path)} resolves to ${reprText(real)}, outside the code root ${reprText(codeRoot)}`,
          );
        }
      }
      return;
    }
    if (MODULE_REFERENCE_RE.test(code)) {
      throw new RangeError(
        `\`code:\` value ${reprText(code)} looks like a Python \`module:function\` reference; use \`./file.ts:fn\``,
      );
    }
    await compileBody(this.id, code);
  }

  /**
   * Run the code on the bound input record and return its one value.
   *
   * The built-in identity returns the record's field as bound. A reference's function or an
   * inline body runs on a worker thread and gets `inputs`, a frozen host copy (ints as
   * `number`s, big ones as `bigint`s) behind a proxy that throws on an undeclared name; what it
   * throws rejects as a `CodeError` (an inline body's frames in its stack named after
   * `sourceFile` and the body line), its `console` calls go to this thread's `console`, and its
   * result goes through host intake (a top-level `return inputs` returns the record). The worker
   * runs with the caller's `current()` values and goes back to the idle pool after its reply.
   *
   * @param caps.signal Aborting it terminates the worker, then rejects with its reason. A body
   *   blocked in a synchronous system call (`execSync`, a blocking read) ends when that call
   *   returns; the rejection waits for it, so a stop never settles while a command it ran still runs.
   * @throws ReferenceError when the built-in's field isn't in the record.
   * @throws The signal's reason when `caps.signal` is aborted before or while the code runs.
   * @throws CodeError when a reference's module throws while it loads.
   * @throws TypeError when a reference's module has no function export by that name.
   * @throws TypeStripError when the body doesn't parse or strip.
   * @throws SyntaxError when the function constructor refuses the stripped body.
   * @throws CodeError when the body or the referenced function throws, also later from a timer
   *   or an unhandled rejection.
   * @throws Error when the code returns `undefined`, or the worker exits before it replies (a
   *   `process.exit`, a body awaiting a promise nothing can resolve) or fails to start.
   * @throws HostOutputError when host intake refuses the result, or the result nests too deep to
   *   copy back from the worker (`contract.output`).
   */
  override async forward(inputs: Record<string, unknown>, caps: NodeCaps = {}): Promise<Output> {
    const code = this.code;
    if (typeof code !== "string") {
      if (!Object.hasOwn(inputs, code.field)) throw undeclaredInput(code.field);
      return new Output(inputs[code.field]);
    }
    caps.signal?.throwIfAborted();
    const reply = await runInWorker(
      {
        nodeId: this.id,
        mode: this.mode as "reference" | "inline",
        code,
        baseDir: this.baseDir,
        inputs: packFiles(toHost(inputs as Value)),
        defaults: current(),
      },
      caps.signal,
    );
    if (!reply.ok) {
      const error = replyError(reply);
      // The failure's traceback shows this stack; `original` keeps the raw frames `bodyLine` reads.
      if (error instanceof CodeError && typeof error.stack === "string") {
        error.stack = labelBodyFrames(error.stack, this.id, this.sourceFile);
      }
      throw error;
    }
    return new Output(hostOutput(unpackFiles(reply.value)));
  }
}
