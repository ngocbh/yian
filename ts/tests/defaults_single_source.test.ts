/**
 * A port of `tests/test_defaults_single_source.py`: every default and limit is defined only in
 * `defaults.ts`.
 *
 * The scan reads `ts/src` as text, with comments and strings blanked by `blank`, and never
 * imports it.
 *
 * Ported: the shape self-test, the scan of `ts/src`, the stale-allowlist and allowed-files
 * checks, the inline-cap-guard shapes, and the OAuth-callback and question-count cap rows.
 * Changed: the same rule holds in TS (decision 4b), so a small token scanner replaces Python's
 * `ast` walk and the allowlist names TS sites. A flagged literal is an integer of 2 or more or a
 * negative integer (decimal, hex, binary, octal, separators, bigint; floats are not flagged, as
 * in Python), anywhere in: a top-level `const`/`let`/`var` value, including inside top-level
 * blocks; a class field value; a parameter default, including inside a destructured parameter;
 * and, as the TS forms of a keyword default, a destructuring default in a declaration
 * (`const { timeoutMs = 30_000 } = options`) and a numeric constant right of `??` or `||`.
 * `len(...)` comparisons become `.length` comparisons. The self-test's Python-only shapes
 * (positional-only parameters, a `with` block, a nested class) have no TS form and are left out
 * of its source. The scanner assumes class members and
 * statements end in `;` (nothing in `ts/src` relies on automatic semicolon insertion). It is not a
 * real parser because the `typescript` 7 package has no stable JS AST API and no other parser is
 * a dependency.
 */

import { expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { blank } from "./engine/source_scan.ts";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

// Each `(path relative to ts/src, name)` the scan finds that is not a default or limit, with
// the reason, keyed `path: name`. A method parameter is `Class.method.parameter`, a fallback
// `function.??` or `function.||`, and an unnamed arrow function `<arrow>`.
const ALLOWED: Record<string, string> = {
  // Format versions and fixed codes.
  "runtime/log.ts: SCHEMA_VERSION": "run-record schema version",
  "runtime/log.ts: RETIRED_SCHEMAS": "schema versions this version refuses by name",
  "memory/episodic.ts: SNAPSHOT_VERSION": "memory snapshot format version",
  "memory/reflective.ts: SNAPSHOT_VERSION": "memory snapshot format version",
  "memory/storage/sqlite.ts: VERSION": "sqlite store format version",
  "runtime/run_index.ts: INDEX_VERSION": "run index layout version",
  "memory/storage/sqlite.ts: SQLITE_BUSY": "SQLite result code",
  "memory/storage/sqlite.ts: SQLITE_LOCKED": "SQLite result code",
  "memory/storage/sqlite.ts: SQLITE_CORRUPT": "SQLite result code",
  "memory/storage/sqlite.ts: SQLITE_NOTADB": "SQLite result code",
  "platform.ts: SQLITE_BUSY": "SQLite result code",
  "platform.ts: SQLITE_LOCKED": "SQLite result code",
  "platform.ts: SQLITE_CANTOPEN": "SQLite result code",
  "platform.ts: SQLITE_NOTADB": "SQLite result code",
  "platform.ts: primaryCode.??": "no SQLite result code, which matches no primary code",
  "runtime/run_index.ts: SQLITE_BUSY": "SQLite result code",
  "runtime/run_index.ts: busyOr.??": "no SQLite result code, which matches no primary code",
  "compile/validation.ts: RELATION_RANK": "ordering of type relations",
  "expr/grammar.ts: LEVEL": "operator precedence levels",
  "tools/mcp.ts: INVALID_REQUEST": "JSON-RPC error code",
  "tools/mcp.ts: INVALID_REQUEST_CODES": "JSON-RPC error codes",
  "llm/framing.ts: LF": "the line feed byte",
  "llm/framing.ts: ASCII_SPACE": "the ASCII whitespace bytes",
  "llm/testing.ts: NULL_BODY_STATUSES": "HTTP statuses whose response has no body",
  // Display and message formatting.
  "cli/run.ts: ERR_CONTEXT": "lines of YAML shown around an error",
  "cli/run.ts: flowBanner.||": "terminal width when the output reports none",
  "cli/run.ts: SGR": "terminal color and style codes",
  "cli/run.ts: SPIN_MS": "spinner frame interval",
  "cli/terminal.ts: Terminal.#runWidget.<arrow>.||": "terminal width when the output reports none",
  "cli/terminal.ts: writeRule.||": "terminal width when the output reports none",
  "compose/validate/case_branches.ts: SHOWN_LIMIT": "width of a value shown in a message",
  "run.ts: LITERAL_LABEL_WIDTH": "width of a literal shown in a type label",
  "llm_clients/redact.ts: MIN_VISIBLE": "key characters a masked echo must show to count",
  "llm/errors.ts: MAX_DETAIL_CHARS": "length of provider error text kept in a message",
  "llm/testing.ts: MAX_DIFFS": "differences listed in a cassette mismatch message",
  "tools/mcp.ts: STDERR_LINES": "MCP server stderr lines shown in a start error",
  "tools/mcp.ts: STDERR_KEEP_CHARS": "MCP server stderr kept for a start error",
  "diagnostics.ts: getCloseMatches.n": "suggestions offered, difflib's own default",
  // Internal sizes that no user input reaches.
  "expr/lineage.ts: MAX_SNAPSHOT_DEPTH": "diagnostic retention depth, internal",
  "llm/executor.ts: MAX_ERROR_BODY": "bytes of an error response read, internal",
  "runtime/log.ts: READ_CHUNK_BYTES": "bytes of a run log read at a time, internal",
  "llm/testing.ts: MIN_BASE64_FRAGMENT": "shortest base64 fragment a cassette check matches, internal",
  "nodes/code/worker.ts: COPY_DEPTH": "depth a CODE result is copied back from its worker, above any value limit",
  // Math, language and host constants.
  "expr/expressions.ts: EXACT_LIMIT": "2^53, past which a float is not exact",
  "typesys/json.ts: MAX_SAFE": "the largest safe integer",
  "typesys/json.ts: byCodeUnits": "a comparator's -1 result",
  "tools/mcp.ts: MAX_TIMER_MS": "the longest delay setTimeout takes",
  "platform.ts: SLEEP_CHUNK_MS": "longest timer armed at once; longer sleeps run in chunks",
  "llm/errors.ts: parseRetryAfter.now": "milliseconds per second",
  "typesys/temporal.ts: UNIT_SECONDS": "seconds per duration unit",
  "cli/index.ts: args": "process.argv entries before the arguments",
  // Host call options and network tuning, passed per call.
  "run.ts: startRun.numWorkers": "host run option, passed per call",
  "llm_clients/auth/openai.ts: OpenAIAuth.constructor.requestTimeoutMs": "network timeout",
  "llm_clients/auth/openai.ts: OpenAIAuth.browserLogin.timeoutMs": "sign-in timeout",
  "llm_clients/auth/openai.ts: OpenAIAuth.deviceLogin.timeoutMs": "sign-in timeout",
  "llm_clients/auth/store.ts: AuthStore.constructor.lockTimeoutMs": "file lock timeout",
  "runtime/run_index.ts: DEFAULT_TIMEOUT_MS": "run index lock timeout, a host call's timeoutMs",
  "nodes/contracts.ts: DEFAULT_MAX_EVENT_BYTES": "event cap of a NodeSteps built outside a run; the engine passes the run's recorded limit",
  "llm_clients/auth/store.ts: AuthStore.constructor.refreshMarginMs": "time before token expiry to refresh",
  "llm_clients/model_list.ts: MODEL_LIST_TIMEOUT_SECONDS": "network timeout",
  "tools/mcp.ts: MAX_PAGES": "tool-list pages read from an MCP server, as the SDK's cap",
  "tools/mcp.ts: TERMINATE_SESSION_MS": "wait for an HTTP MCP server to end its session on close",
  "platform.ts: LOCK_POLL_MS": "sleep between tries on a busy file lock",
  // Provider contracts.
  "llm/errors.ts: MAX_RETRY_AFTER_SECONDS": "provider retry-after cap, as openai's (Python's is a float its scan skips)",
  "llm/errors.ts: OVERFLOW_RETRY_AFTER": "one past the retry-after cap, so an overflowing delay is refused",
  "llm/executor.ts: MAX_RETRY_DELAY_SECONDS": "longest model retry backoff, as openai's (Python writes it inline in _delay)",
  "llm/protocols/anthropic_messages.ts: MAX_TOKENS": "Anthropic's max_tokens per model, a provider contract",
  "llm/protocols/anthropic_messages.ts: DEFAULT_MAX_TOKENS": "Anthropic's max_tokens for other models, a provider contract",
};

// Files whose int literals are all outside the scan's scope.
const ALLOWED_FILES: Record<string, string> = {
  "llm_clients/provider_blocks.ts": "provider API contracts",
};

/** One token of blanked source: its text, kind and offset. */
interface Token {
  text: string;
  kind: "name" | "number" | "string" | "punct";
  index: number;
}

const TOKEN =
  /\s+|([#$_\p{ID_Start}][$\p{ID_Continue}]*)|(0[xXbBoO][\da-fA-F_]+n?|\d[\d_]*(?:\.[\d_]*)?(?:[eE][+-]?\d+)?n?|\.\d[\d_]*(?:[eE][+-]?\d+)?)|("[^"\n]*"|'[^'\n]*')|(\.\.\.|[=!]==|\*\*=|(?:\?\?|\|\||&&|<<|>>>?)=|[<>=!]=|=>|\?\?|\?\.|\|\||&&|\*\*|<<|\+\+|--|[-+*/%&|^]=|[^])/gu;

/** Return the tokens of `code`, which `blank` has already cleared of comments and strings. */
function tokenize(code: string): Token[] {
  const tokens: Token[] = [];
  for (const match of code.matchAll(TOKEN)) {
    const kind = match[1] ? "name" : match[2] ? "number" : match[3] ? "string" : match[4] ? "punct" : null;
    if (kind) tokens.push({ text: match[0], kind, index: match.index });
  }
  return tokens;
}

/** Words after which `-` is a sign and before which no value ends. */
const KEYWORDS = new Set([
  "async",
  "await",
  "case",
  "delete",
  "do",
  "else",
  "export",
  "function",
  "in",
  "instanceof",
  "new",
  "of",
  "return",
  "throw",
  "typeof",
  "void",
  "yield",
]);
const CONTROL = new Set(["if", "for", "while", "switch", "catch", "with"]);
const DECLARATIONS = new Set(["const", "let", "var"]);
const STATEMENT_MODIFIERS = new Set(["export", "declare", "default"]);
const MEMBER_MODIFIERS = new Set([
  "public",
  "private",
  "protected",
  "readonly",
  "static",
  "declare",
  "override",
  "accessor",
  "abstract",
]);
const OPENERS = new Set(["(", "[", "{"]);
const CLOSERS = new Set([")", "]", "}"]);

/** Return the int an integer literal means, or `null` for a float or a non-number. */
function intValue(token: Token | undefined): bigint | null {
  if (token?.kind !== "number" || !/^(0[xXbBoO][\da-fA-F_]+|\d[\d_]*)n?$/.test(token.text)) return null;
  return BigInt(token.text.replace(/_/g, "").replace(/n$/, ""));
}

/** Whether `token` ends a value, so a `-` after it subtracts. */
function endsValue(token: Token | undefined): boolean {
  if (token === undefined) return false;
  if (token.kind === "name") return !KEYWORDS.has(token.text);
  return token.kind !== "punct" || token.text === ")" || token.text === "]";
}

/** Whether `tokens[i]` is an int literal of at least 2, or a negative int literal. */
function isFlaggedInt(tokens: readonly Token[], i: number): boolean {
  const value = intValue(tokens[i]);
  if (value === null) return false;
  return value >= 2n || (value > 0n && tokens[i - 1]?.text === "-" && !endsValue(tokens[i - 2]));
}

/** Return the index of the bracket closing the one at `open`, or `tokens.length`. */
function matching(tokens: readonly Token[], open: number): number {
  let depth = 0;
  for (let k = open; k < tokens.length; k++) {
    if (OPENERS.has(tokens[k]!.text)) depth++;
    else if (CLOSERS.has(tokens[k]!.text) && --depth === 0) return k;
  }
  return tokens.length;
}

/**
 * Return the index of the `{` or `=>` that follows the `)` at `close`, past an optional return
 * type, when the parentheses hold a signature; else -1.
 */
function bodyAfter(tokens: readonly Token[], close: number): number {
  let k = close + 1;
  if (tokens[k]?.text === ":") {
    let depth = 0;
    for (k++; k < tokens.length; k++) {
      const text = tokens[k]!.text;
      // A `{` after a complete type opens the body; after `:`, `|` or `<` it opens an object type.
      if (depth === 0 && (text === "=>" || (text === "{" && /^[\w$"')\]}>]/.test(tokens[k - 1]!.text)))) break;
      if (OPENERS.has(text) || text === "<") depth++;
      else if ((CLOSERS.has(text) || text === ">") && --depth < 0) return -1;
      else if (depth === 0 && (text === ";" || text === "," || text === "=")) return -1;
    }
  }
  return tokens[k]?.text === "{" || tokens[k]?.text === "=>" ? k : -1;
}

/** Return the name of the function whose parameters open at `open`, or `<arrow>`. */
function functionName(tokens: readonly Token[], open: number): string {
  let p = open - 1;
  if (tokens[p]?.text === ">") {
    // Type parameters: `name<T>(`.
    for (let depth = 0; p >= 0; p--) {
      if (tokens[p]!.text === ">") depth++;
      else if (tokens[p]!.text === "<" && --depth === 0) break;
    }
    p--;
  }
  const before = tokens[p];
  if (before?.kind === "name" && !KEYWORDS.has(before.text)) return before.text;
  while (tokens[p] && ["async", "function", "*"].includes(tokens[p]!.text)) p--;
  // An arrow or function expression takes the name it is bound to: `name = (` or `name: (`.
  if ((tokens[p]?.text === "=" || tokens[p]?.text === ":") && tokens[p - 1]?.kind === "name") return tokens[p - 1]!.text;
  return "<arrow>";
}

/** Return the name of the class whose body opens at `open`, or `null` when it isn't one. */
function className(tokens: readonly Token[], open: number): string | null {
  for (let p = open - 1; p >= 0; p--) {
    const text = tokens[p]!.text;
    if (text === "class") {
      const name = tokens[p + 1];
      return name?.kind === "name" && name.text !== "extends" && name.text !== "implements" ? name.text : "<class>";
    }
    if ([";", "{", "}", "(", ")", "=>"].includes(text)) return null;
  }
  return null;
}

const ARITHMETIC = new Map<string, (a: bigint, b: bigint) => bigint>([
  ["<<", (a, b) => a << b],
  ["+", (a, b) => a + b],
  ["-", (a, b) => a - b],
  ["*", (a, b) => a * b],
]);
const LEVELS = [["<<"], ["+", "-"], ["*"]];

/** Return the int a constant expression of int literals evaluates to, else `null`. */
function constantInt(tokens: readonly Token[]): bigint | null {
  let at = 0;
  const atom = (): bigint | null => {
    if (tokens[at]?.text === "-") {
      at++;
      const operand = atom();
      return operand === null ? null : -operand;
    }
    let value: bigint | null;
    if (tokens[at]?.text === "(") {
      at++;
      value = level(0);
      if (tokens[at++]?.text !== ")") return null;
    } else {
      value = intValue(tokens[at++]);
    }
    if (value === null || tokens[at]?.text !== "**") return value;
    at++;
    const exponent = atom();
    return exponent === null ? null : value ** exponent;
  };
  const level = (n: number): bigint | null => {
    if (n === LEVELS.length) return atom();
    let left = level(n + 1);
    while (left !== null && LEVELS[n]!.includes(tokens[at]?.text ?? "")) {
      const op = ARITHMETIC.get(tokens[at++]!.text)!;
      const right = level(n + 1);
      left = right === null ? null : op(left, right);
    }
    return left;
  };
  try {
    const value = level(0);
    return at === tokens.length ? value : null;
  } catch {
    // A negative exponent or a shift too large to compute.
    return null;
  }
}

/** Whether `token` can belong to a constant expression of int literals. */
const inConstant = (token: Token): boolean =>
  intValue(token) !== null || ["(", ")", "-", "+", "*", "**", "<<"].includes(token.text);

/** Tokens that may end an operand, so what precedes them is the whole operand. */
const OPERAND_ENDS = new Set([";", ",", ")", "]", "}", ":", "?", "??", "||", "&&"]);

/**
 * Return the tokens of the constant operand starting at `start`, or `null` when the operand
 * there is not a constant expression of int literals.
 */
function constantOperand(tokens: readonly Token[], start: number): Token[] | null {
  let end = start;
  for (let depth = 0; end < tokens.length && inConstant(tokens[end]!); end++) {
    if (tokens[end]!.text === "(") depth++;
    else if (tokens[end]!.text === ")" && --depth < 0) break;
  }
  const next = tokens[end];
  const run = tokens.slice(start, end);
  return (next === undefined || OPERAND_ENDS.has(next.text)) && constantInt(run) !== null ? run : null;
}

type Kind = "block" | "class" | "function" | "params" | "pattern" | "other";

/** An open bracket while scanning. */
interface Frame {
  kind: Kind;
  /** The class or function name, for `class`, `function` and `params` frames. */
  name: string;
  /** Index of the closing bracket. */
  close: number;
  /** Index of the first token of the current statement, member or element. */
  start: number;
  /** Whether that statement's or member's `=` was seen. */
  assigned: boolean;
}

/**
 * Return the flagged names in one module's source.
 *
 * A top-level declaration (also inside top-level `if`, `try` and loop blocks) or a class field
 * counts when a flagged int appears anywhere in its value; a parameter default or a
 * destructuring default counts in every function. Names are qualified by the enclosing classes
 * and functions, and an arrow or function expression takes the name it is bound to.
 */
function flaggedNames(source: string): Set<string> {
  const tokens = tokenize(blank(source));
  const hits = new Set<string>();
  const stack: Frame[] = [{ kind: "block", name: "", close: tokens.length, start: 0, assigned: false }];
  // The `{` that opens each function body, found with the function's parameters.
  const bodies = new Map<number, string>();
  const qualified = (...names: string[]): string =>
    [...stack.filter((f) => f.kind === "class" || f.kind === "function").map((f) => f.name), ...names].join(".");
  const reset = (frame: Frame, start: number): void => {
    frame.start = start;
    frame.assigned = false;
  };
  const skip = (start: number, words: Set<string>): number => {
    let s = start;
    while (s < tokens.length && words.has(tokens[s]!.text)) s++;
    return s;
  };
  // The end of a value starting at `from`: a `;` (or `,` when `comma`) or a closer at its depth.
  const valueEnd = (from: number, comma: boolean): number => {
    let depth = 0;
    for (let k = from; k < tokens.length; k++) {
      const text = tokens[k]!.text;
      if (OPENERS.has(text)) depth++;
      else if (CLOSERS.has(text) && --depth < 0) return k;
      else if (depth === 0 && (text === ";" || (comma && text === ","))) return k;
    }
    return tokens.length;
  };
  const record = (from: number, end: number, name: string | undefined): void => {
    if (name === undefined) return;
    for (let k = from; k < end; k++) {
      if (isFlaggedInt(tokens, k)) {
        hits.add(name);
        return;
      }
    }
  };
  const nameAt = (k: number): string | undefined => (tokens[k]?.kind === "name" ? tokens[k]!.text : undefined);
  const declares = (frame: Frame): boolean =>
    DECLARATIONS.has(tokens[skip(frame.start, STATEMENT_MODIFIERS)]?.text ?? "");

  const classify = (i: number, top: Frame): Pick<Frame, "kind" | "name"> => {
    const text = tokens[i]!.text;
    const prev = tokens[i - 1]?.text;
    if (text === "(") {
      const control = CONTROL.has(prev ?? "") || (prev === "await" && tokens[i - 2]?.text === "for");
      const body = control ? -1 : bodyAfter(tokens, matching(tokens, i));
      if (body === -1) return { kind: "other", name: "" };
      const name = functionName(tokens, i);
      if (tokens[body]!.text === "{") bodies.set(body, name);
      return { kind: "params", name };
    }
    const body = bodies.get(i);
    if (body !== undefined) return { kind: "function", name: body };
    const atElement = i === top.start || (prev === "..." && i - 1 === top.start);
    if (
      DECLARATIONS.has(prev ?? "") ||
      (top.kind === "params" && atElement) ||
      (top.kind === "pattern" && (atElement || prev === ":"))
    ) {
      return { kind: "pattern", name: "" };
    }
    if (text === "[") return { kind: "other", name: "" };
    if (prev === "=>") return { kind: "function", name: "<arrow>" };
    if (prev === "static" && top.kind === "class") return { kind: "function", name: "static" };
    const cls = className(tokens, i);
    if (cls !== null) return { kind: "class", name: cls };
    const statement =
      prev === ")" ||
      ["else", "try", "finally", "do"].includes(prev ?? "") ||
      ((top.kind === "block" || top.kind === "function") && [undefined, ";", "{", "}"].includes(prev));
    return { kind: statement ? "block" : "other", name: "" };
  };

  for (let i = 0; i < tokens.length; i++) {
    const text = tokens[i]!.text;
    const top = stack.at(-1)!;
    if (i === top.close) {
      stack.pop();
      const parent = stack.at(-1)!;
      // A body's `}` ends its statement or member; so does any bracket in a statement that
      // declares nothing, such as an interface's body.
      const ends = top.kind === "function" || top.kind === "class" || top.kind === "block";
      if ((parent.kind === "block" && (ends || !declares(parent))) || (parent.kind === "class" && top.kind === "function")) {
        reset(parent, i + 1);
      }
    } else if (OPENERS.has(text)) {
      stack.push({ ...classify(i, top), close: matching(tokens, i), start: i + 1, assigned: false });
    } else if (text === ";" && (top.kind === "block" || top.kind === "class")) {
      reset(top, i + 1);
    } else if (text === "," && (top.kind === "params" || top.kind === "pattern")) {
      reset(top, i + 1);
    } else if (text === "??" || text === "||") {
      const operand = constantOperand(tokens, i + 1);
      if (operand?.some((_, k) => isFlaggedInt(tokens, i + 1 + k))) hits.add(qualified(text));
    } else if (text === "=" && (top.kind === "params" || top.kind === "pattern")) {
      let owner = stack.length - 1;
      while (stack[owner]!.kind === "pattern") owner--;
      const name = top.kind === "pattern" ? nameAt(i - 1) : nameAt(skip(top.start, MEMBER_MODIFIERS));
      if (name === undefined) continue;
      const end = valueEnd(i + 1, true);
      record(i + 1, end, stack[owner]!.kind === "params" ? qualified(stack[owner]!.name, name) : qualified(name));
    } else if (text === "=" && top.kind === "class" && !top.assigned) {
      top.assigned = true;
      const name = nameAt(skip(top.start, MEMBER_MODIFIERS));
      record(i + 1, valueEnd(i + 1, false), name === undefined ? undefined : qualified(name));
    } else if (text === "=" && top.kind === "block" && !top.assigned && stack.every((f) => f.kind === "block")) {
      top.assigned = true;
      const keyword = skip(top.start, STATEMENT_MODIFIERS);
      if (DECLARATIONS.has(tokens[keyword]?.text ?? "")) record(i + 1, valueEnd(i + 1, false), nameAt(keyword + 1));
    }
  }
  return hits;
}

/** Return every `.ts` file below `src`, as paths in `/` form. */
function sourcePaths(): string[] {
  return readdirSync(SRC, { recursive: true, encoding: "utf8" })
    .filter((name) => name.endsWith(".ts"))
    .map((name) => name.split(sep).join("/"))
    .sort();
}

let scanned: Set<string> | null = null;

/** Return `path: name` for every flagged binding outside `defaults.ts` and `ALLOWED_FILES`. */
function scan(): Set<string> {
  if (scanned === null) {
    scanned = new Set();
    for (const path of sourcePaths()) {
      if (path === "defaults.ts" || path in ALLOWED_FILES) continue;
      for (const name of flaggedNames(readFileSync(join(SRC, path), "utf8"))) scanned.add(`${path}: ${name}`);
    }
  }
  return scanned;
}

test("scan flags each shape of default", () => {
  const source = `
export const MAX_BYTES = 256 * 1024;
const LIMITS = limits(100, 10);
let maxIntDigits = 4300;
const NO_CAP = -1;
const MASK = 0xffn;
const ONE = 1;
const RATIO = 2.5;
const ENABLED = true;
const LABEL = "100"; // const COMMENTED = 5;
if (ENABLED) {
  const GUARDED = 5;
}
try {
  var TRIED = 6;
} catch {}
for (const item of items) {
  let LOOPED = 1_000;
}

class Model {
  name = field({ maxLength: 64 });
  static retries = 2;
  private readonly width: number = 3;
  #depth = 12;

  method(a: number, b = 3, { c = -1, d = 1 }: Options = {}): void {
    const LOCAL = 99;
    const { timeoutMs = 30_000, flag = false } = options;
    const [first = 2] = list;
    function inner(e = 7) {}
  }
}

function product(n = 256 * 1024, m = 1 << 20, ok = null) {}

async function fetch(timeout = 30.0, attempts = 3) {}

function generic<T>(limit = 10, flag = false): T[] {
  return [];
}

function fallbacks(options: Options) {
  const wait = options.wait ?? 30;
  const width = options.columns || 80;
}

function notFallbacks(options: Options) {
  const one = options.one ?? 1;
  const scaled = options.scale ?? 2 * options.unit;
  const ratio = options.ratio ?? 0.5;
}

const handler = (size: number = 64): number => size;
call(async (depth = -2) => depth);
`;
  expect([...flaggedNames(source)].sort()).toEqual(
    [
      "MAX_BYTES",
      "LIMITS",
      "maxIntDigits",
      "NO_CAP",
      "MASK",
      "GUARDED",
      "TRIED",
      "LOOPED",
      "Model.name",
      "Model.retries",
      "Model.width",
      "Model.#depth",
      "Model.method.b",
      "Model.method.c",
      "Model.method.timeoutMs",
      "Model.method.first",
      "Model.method.inner.e",
      "product.n",
      "product.m",
      "fetch.attempts",
      "generic.limit",
      "fallbacks.??",
      "fallbacks.||",
      "handler",
      "handler.size",
      "<arrow>.depth",
    ].sort(),
  );
});

test("defaults and limits are defined only in yian defaults", () => {
  const unexpected = [...scan()].filter((key) => !(key in ALLOWED)).sort();
  expect(
    unexpected,
    "int defaults or limits defined outside defaults.ts (move them there, or add an ALLOWED entry " +
      "with the reason they are not a default or limit)",
  ).toEqual([]);
});

test("allowlist has no stale entries", () => {
  const found = scan();
  expect(
    Object.keys(ALLOWED)
      .filter((key) => !found.has(key))
      .sort(),
    "ALLOWED entries the scan no longer finds",
  ).toEqual([]);
});

test("allowed files exist", () => {
  const missing = Object.keys(ALLOWED_FILES)
    .filter((name) => !existsSync(join(SRC, name)))
    .sort();
  expect(missing, "ALLOWED_FILES entries that no longer exist").toEqual([]);
});

const COMPARISONS = new Set(["<", ">", "<=", ">=", "==", "!=", "===", "!=="]);
const OPERAND_BOUNDS = new Set([...OPERAND_ENDS, ...COMPARISONS, "=", "=>", "return"]);

/** Return the operand next to the comparison at `at`, before it when `step` is -1. */
function comparedOperand(tokens: readonly Token[], at: number, step: 1 | -1): Token[] {
  const operand: Token[] = [];
  const [opens, closes] = step === 1 ? [OPENERS, CLOSERS] : [CLOSERS, OPENERS];
  for (let k = at + step, depth = 0; k >= 0 && k < tokens.length; k += step) {
    const text = tokens[k]!.text;
    if (opens.has(text)) depth++;
    else if (closes.has(text) && --depth < 0) break;
    else if (depth === 0 && OPERAND_BOUNDS.has(text)) break;
    operand.push(tokens[k]!);
  }
  return step === 1 ? operand : operand.reverse();
}

/** Return whether `source` compares a `.length` against a constant `value`. */
function comparesLengthTo(source: string, value: bigint): boolean {
  const tokens = tokenize(blank(source));
  const isLength = (operand: Token[]): boolean =>
    operand.length >= 3 && operand.at(-1)!.text === "length" && [".", "?."].includes(operand.at(-2)!.text);
  return tokens.some((token, i) => {
    if (!COMPARISONS.has(token.text)) return false;
    const left = comparedOperand(tokens, i, -1);
    const right = comparedOperand(tokens, i, 1);
    return (isLength(left) && constantInt(right) === value) || (isLength(right) && constantInt(left) === value);
  });
}

test("inline cap guards flag each shape", () => {
  expect(constantInt(tokenize("16 * 1024"))).toBe(16384n);
  expect(constantInt(tokenize("1 << 14"))).toBe(16384n);
  expect(constantInt(tokenize("2 ** 14"))).toBe(16384n);
  expect(constantInt(tokenize("n * 1024"))).toBeNull();
  expect(comparesLengthTo("if (raw.length > 4) {}", 4n)).toBe(true);
  expect(comparesLengthTo("const ok = 1 <= raw.length && raw.length <= 2 + 2;", 4n)).toBe(true);
  expect(comparesLengthTo("const ok = 1 <= raw.length && raw.length <= most;", 4n)).toBe(false);
});

test("oauth callback cap is not inline", () => {
  const source = readFileSync(join(SRC, "llm_clients/auth/openai.ts"), "utf8");
  const tokens = tokenize(blank(source));
  const inline = new Set<number>();
  // Every run of constant tokens, and every slice of it that is an expression.
  for (let start = 0; start < tokens.length; start++) {
    for (let end = start + 1; end <= tokens.length && inConstant(tokens[end - 1]!); end++) {
      if (constantInt(tokens.slice(start, end)) === 16384n) {
        inline.add(source.slice(0, tokens[start]!.index).split("\n").length);
      }
    }
  }
  expect([...inline], "OAuth callback cap written inline at these lines").toEqual([]);
});

test("question count cap is not inline", () => {
  expect(comparesLengthTo(readFileSync(join(SRC, "nodes/human_input/questions.ts"), "utf8"), 4n)).toBe(false);
});
