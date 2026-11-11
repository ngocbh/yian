/**
 * Authored type text → `TypeExpr` AST → resolved `Type`.
 *
 * Parses an authored type expression into a `TypeExpr` AST, the bridge from the authoring
 * contract to the typed value system (`values.ts`), and prints a parsed type back in the
 * parenthesized spelling. Compose and compile type resolution consume it; runtime receives
 * only resolved `Type` values.
 */

import { current } from "../defaults.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import {
  type Bound,
  type Bounds,
  BUILTIN_CONSTANTS,
  boundText,
  canonicalUnionOrder,
  IDENTIFIER_PATTERN,
  inboxType,
  isMemoryType,
  listKindForElement,
  literalLabel,
  memoryType,
  NAME_PATTERN,
  Type,
  TypeCheckError,
  typeEquals,
  typeLevels,
  typesOverlap,
  unionTagFields,
  ValueKind,
  validateNumericBounds,
} from "./values.ts";

// --------------------------------------------------------------------------------------- //
// TypeExpr AST
// --------------------------------------------------------------------------------------- //

/** A scalar keyword as a parsed type names it (the lowercase `none` reads as `None`). */
export type ScalarName =
  | "str"
  | "int"
  | "float"
  | "bool"
  | "date"
  | "datetime"
  | "object"
  | "file"
  | "None"
  | "Any";

/** A scalar type; `int` and `float` may carry an inclusive range. */
export interface ScalarExpr {
  readonly kind: "scalar";
  readonly name: ScalarName;
  readonly bounds?: Bounds;
}

/** `list(X)`. */
export interface ListExpr {
  readonly kind: "list";
  readonly element: TypeExpr;
}

/** A registry name (a record, variant or alias typedef). */
export interface RefExpr {
  readonly kind: "ref";
  readonly name: string;
}

/** `Optional(X)`: the resolved type is X's, marked nullable. */
export interface OptionalExpr {
  readonly kind: "optional";
  readonly inner: TypeExpr;
}

/** An inline `Literal(...)` enum (a tag-only variant), its tags in authored order. */
export interface EnumExpr {
  readonly kind: "enum";
  readonly tags: readonly string[];
}

/**
 * `dict(K, V)`: resolves to a lenient object. Its key and value types, when read, are kept only to
 * check the names they use and to refuse an inbox inside it.
 */
export interface DictExpr {
  readonly kind: "dict";
  readonly args?: readonly TypeExpr[];
}

/** The authored `Union(...)` members, in authored order. */
export interface UnionExpr {
  readonly kind: "union";
  readonly members: readonly TypeExpr[];
}

/** `Inbox(X)`: a handle on a top-level inbox whose events are `X`s. */
export interface InboxExpr {
  readonly kind: "inbox";
  readonly element: TypeExpr;
}

/** `FileMemory` or `VectorDBMemory(X)`: a built-in memory type, with its item record if any. */
export interface MemoryExpr {
  readonly kind: "memory";
  readonly type: string;
  readonly item: TypeExpr | null;
}

/** A parsed type expression, still unresolved against any registry. */
export type TypeExpr =
  | ScalarExpr
  | ListExpr
  | RefExpr
  | OptionalExpr
  | EnumExpr
  | DictExpr
  | UnionExpr
  | InboxExpr
  | MemoryExpr;

const SCALAR_KEYWORDS: ReadonlySet<string> = new Set([
  "str", "int", "float", "bool", "date", "datetime", "object", "file", "None", "none", "Any",
]);

// Names a typedef must not shadow (typing constructors and the generic builtins).
const TYPING_CONSTRUCTORS: ReadonlySet<string> = new Set([
  "Optional", "Literal", "Union", "List", "Dict", "list", "dict", "Any", "Inbox",
]);

/** The built-in record type names every flow may use and no typedef may take. */
const BUILTIN_RECORD_NAMES: ReadonlySet<string> = new Set(["Message"]);

/**
 * The built-in memory types, which no typedef may take. `VectorDBMemory` may name its item
 * record, as in `VectorDBMemory(Case)`; `FileMemory` takes none.
 */
export const BUILTIN_MEMORY_TYPES: ReadonlySet<string> = new Set(["FileMemory", "VectorDBMemory"]);

/**
 * Names a type expression may start with besides typedefs, derived from the keyword sets
 * so they cannot drift.
 */
export const BUILTIN_TYPE_NAMES: ReadonlySet<string> = new Set([
  ...SCALAR_KEYWORDS,
  ...TYPING_CONSTRUCTORS,
  ...BUILTIN_RECORD_NAMES,
  ...BUILTIN_MEMORY_TYPES,
  "any",
]);

/** Each scalar name's storage kind. */
const SCALAR_KIND: ReadonlyMap<ScalarName, ValueKind> = new Map([
  ["str", ValueKind.STRING],
  ["int", ValueKind.INTEGER],
  ["float", ValueKind.NUMBER],
  ["bool", ValueKind.BOOLEAN],
  ["date", ValueKind.DATE],
  ["datetime", ValueKind.DATETIME],
  ["object", ValueKind.OBJECT],
  ["None", ValueKind.NONE],
  ["file", ValueKind.FILE],
  ["Any", ValueKind.ANY],
]);

/** Return whether `text` encodes as UTF-8, i.e. holds no lone Unicode surrogate. */
export function encodesAsUtf8(text: string): boolean {
  return text.isWellFormed();
}

/** Whether typedef name `name` shadows a scalar keyword or a typing constructor. */
function isShadow(name: string): boolean {
  return (
    SCALAR_KEYWORDS.has(name) ||
    SCALAR_KEYWORDS.has(name.toLowerCase()) ||
    TYPING_CONSTRUCTORS.has(name)
  );
}

/**
 * Parse an authored type expression into a `TypeExpr` AST.
 *
 * The vocabulary is `str`/`int`/`float`/`bool`/`Any`/`date`/`datetime`/`object`/`file`/
 * `None`, where `Any` is any JSON value except null and `object` is a JSON object,
 * inclusive ranges `int(1..12)`/`float(0.0..1.0)` (either end may be open), and the
 * constructors `list(X)`/`Optional(X)`/`dict(K, V)`/`Literal(a, "b c")`/
 * `Union(A, B, ...)`/`Inbox(X)`, and the memory types `FileMemory` and `VectorDBMemory`/
 * `VectorDBMemory(X)`. A bare name that is neither a scalar, a constructor nor a memory type is
 * a registry reference. `Union(...)` members are only checked to be records when the type is
 * resolved. A quoted `Literal` member takes JSON escapes.
 *
 * Example: `parseType("Union(Buy, Sell)")` gives
 * `{kind: "union", members: [{kind: "ref", name: "Buy"}, {kind: "ref", name: "Sell"}]}`.
 *
 * @throws TypeCheckError when the text is not a valid expression (including unbalanced
 *   parentheses, a lone Unicode surrogate, or an empty `Literal` member), uses an unknown
 *   constructor, the lowercase `any` or the retired `topics`, writes a constructor without its arguments, gives a
 *   numeric range with no end, no `..`, or as two arguments, spells a union as `A | B` (the hint suggests `Union(A, B)`), lists
 *   `None` as a `Union(...)` member (the hint suggests `Optional(Union(...))`), wraps an
 *   `Inbox(...)` in `Optional(...)`, or holds a `[` outside a quoted string (the message shows
 *   the parenthesized spelling).
 */
export function parseType(text: string): TypeExpr {
  const src = text.trim();
  if (hasUnquotedBracket(src)) throw bracketError(src);
  checkEncodable(src);
  return new TypeParser(src).parse();
}

/** Reject type text `src` holding a lone Unicode surrogate. */
function checkEncodable(src: string): void {
  if (!encodesAsUtf8(src)) {
    throw new TypeCheckError(
      `type expression ${reprText(src)} contains a Unicode surrogate ` +
        "(types must be valid UTF-8 text)",
    );
  }
}

/** The `malformed type expression` error for `src`, with an optional `detail`. */
function malformedError(src: string, detail = ""): TypeCheckError {
  const suffix = detail ? `: ${detail}` : "";
  return new TypeCheckError(`malformed type expression ${reprText(src)}${suffix}`);
}

/** Explain that types use parentheses, showing `src` in that spelling when it parses. */
function bracketError(src: string): TypeCheckError {
  let spelled: string;
  try {
    spelled = legacyRespelling(src);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    return new TypeCheckError(`types use parentheses, not brackets (${error.message})`);
  }
  return new TypeCheckError(`types use parentheses: \`${spelled}\``);
}

/**
 * Spell bracket-style type text in the parenthesized spelling, for error messages.
 *
 * A `dict[K, V]` becomes `object`, the lenient mapping it stood for.
 *
 * Example: `legacyRespelling("Optional[str]")` gives `"Optional(str)"`.
 *
 * @throws TypeCheckError when `text` is not a type in the bracket spelling either.
 */
export function legacyRespelling(text: string): string {
  return formatTypeExpr(dictAsObject(parseLegacyType(text)));
}

/** Replace every `dict(K, V)` in a parsed type with `object`. */
function dictAsObject(t: TypeExpr): TypeExpr {
  switch (t.kind) {
    case "dict":
      return { kind: "scalar", name: "object" };
    case "list":
      return { kind: "list", element: dictAsObject(t.element) };
    case "inbox":
      return { kind: "inbox", element: dictAsObject(t.element) };
    case "optional":
      return { kind: "optional", inner: dictAsObject(t.inner) };
    case "union":
      return { kind: "union", members: t.members.map(dictAsObject) };
    default:
      return t;
  }
}

/** Whether `src` holds a `[` outside quoted strings, i.e. is in the bracket spelling. */
function hasUnquotedBracket(src: string): boolean {
  let quote: string | null = null;
  let escaped = false;
  for (const ch of src) {
    if (quote !== null) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") {
      quote = ch;
    } else if (ch === "[") {
      return true;
    }
  }
  return false;
}

// An escape is a backslash and any character but LF; a regex `.` would
// also refuse CR, U+2028 and U+2029.
const QUOTED = String.raw`"(?:[^"\\]|\\[^\n])*"|'(?:[^'\\]|\\[^\n])*'`;

// NUMBER needs a digit after a dot, so `0..1` reads as `0`, `..`, `1`. Its digits are
// ASCII only, so `int(١..٣)` is not a range.
const TYPE_TOKEN = new RegExp(
  String.raw`\s*(?:(?<number>[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)` +
    `|(?<name>${NAME_PATTERN})|(?<string>${QUOTED})` +
    String.raw`|(?<punct>\.\.|[(),|]))`,
  "uy",
);

const TOKEN_KINDS = ["number", "name", "string", "punct"] as const;

type TokenKind = (typeof TOKEN_KINDS)[number];

interface Token {
  readonly kind: TokenKind;
  readonly text: string;
}

/** Split `src` into `pattern`'s tokens, or return `null` where no token matches. */
function tokenize(pattern: RegExp, src: string): Token[] | null {
  const tokens: Token[] = [];
  for (let pos = 0; pos < src.length; pos = pattern.lastIndex) {
    pattern.lastIndex = pos;
    const groups = pattern.exec(src)?.groups;
    if (groups === undefined) return null;
    const kind = TOKEN_KINDS.find((name) => groups[name] !== undefined)!;
    tokens.push({ kind, text: groups[kind]! });
  }
  return tokens;
}

/** Recursive-descent parser for the parenthesized type grammar. */
class TypeParser {
  private readonly src: string;
  private readonly tokens: Token[];
  private index = 0;
  private depth = 0;
  private readonly maxDepth = current().maxTypeNesting;

  constructor(src: string) {
    this.src = src;
    const tokens = tokenize(TYPE_TOKEN, src);
    if (tokens === null) {
      // An unclosed quote often means a `{...}` mapping's comma split the type.
      if (this.opensUnclosed()) this.unbalanced();
      this.malformed();
    }
    this.tokens = tokens;
  }

  parse(): TypeExpr {
    const t = this.type();
    if (this.peek() === ")") this.unbalanced();
    if (this.peek() !== null) this.malformed();
    return t;
  }

  /** The next token's text, or `null` at the end. */
  private peek(): string | null {
    return this.tokens[this.index]?.text ?? null;
  }

  private peekKind(): TokenKind | null {
    return this.tokens[this.index]?.kind ?? null;
  }

  private next(): Token {
    const token = this.tokens[this.index];
    if (token === undefined) this.ended();
    this.index += 1;
    return token;
  }

  private expect(text: string): void {
    if (this.peek() === null && text === ")") this.unbalanced();
    if (this.next().text !== text) this.malformed();
  }

  private malformed(detail = ""): never {
    throw malformedError(this.src, detail);
  }

  /** Whether the text opens more parentheses than it closes. */
  private opensUnclosed(): boolean {
    return this.src.split("(").length > this.src.split(")").length;
  }

  /** Fail for text that ends before the type does. */
  private ended(): never {
    if (this.opensUnclosed()) this.unbalanced();
    this.malformed();
  }

  /** Reject a `,` directly before `)`. */
  private trailingComma(ctor: string): void {
    if (this.peek() === ")") this.malformed(`remove the trailing comma in ${ctor}(...)`);
  }

  private unbalanced(): never {
    this.malformed(
      "unbalanced parentheses — if this type sits inside a `{...}` mapping, its comma " +
        "split it; quote it or use block style",
    );
  }

  private type(): TypeExpr {
    const t = this.primary();
    if (this.peek() === "|") {
      const operands = [t];
      while (this.peek() === "|") {
        this.next();
        operands.push(this.primary());
      }
      const names = operands.map(formatTypeExpr).join(", ");
      this.malformed(`use Union(${names}) (| is not a type operator)`);
    }
    return t;
  }

  private primary(): TypeExpr {
    const { kind, text: name } = this.next();
    // `None` is the null type; `True` and `False` are no types.
    if (kind !== "name" || (BUILTIN_CONSTANTS.has(name) && name !== "None")) this.malformed();
    if (this.peek() !== "(") return namedType(name);
    this.next();
    this.depth += 1;
    if (this.depth > this.maxDepth) throw nestedTooDeeply(this.src, this.maxDepth);
    const t = this.constructed(name);
    this.depth -= 1;
    return t;
  }

  /** Read the arguments of constructor `name` after its `(`. */
  private constructed(name: string): TypeExpr {
    switch (name) {
      case "int":
      case "float":
        return this.range(name);
      case "list":
      case "List":
        return { kind: "list", element: this.oneType(name) };
      case "Optional": {
        const inner = this.oneType(name);
        if (inner.kind === "inbox") {
          throw new TypeCheckError(
            `an inbox is never optional in ${reprText(this.src)}; write Inbox(...)`,
          );
        }
        return { kind: "optional", inner };
      }
      case "Inbox":
        return { kind: "inbox", element: this.oneType(name) };
      case "VectorDBMemory":
        return { kind: "memory", type: name, item: this.oneType(name) };
      case "FileMemory":
        throw new TypeCheckError(`FileMemory takes no item type in ${reprText(this.src)}`);
      case "Literal":
        return { kind: "enum", tags: this.literalTags() };
      case "dict":
      case "Dict":
        return { kind: "dict", args: this.args(name, 2) };
      case "Union": {
        const members = this.args(name, null);
        if (members.some((m) => m.kind === "scalar" && m.name === "None")) {
          throw new TypeCheckError(
            `None is not a union member in ${reprText(this.src)}; use Optional(Union(...))`,
          );
        }
        return { kind: "union", members };
      }
    }
    throw new TypeCheckError(
      `unknown type constructor ${reprText(name)} in ${reprText(this.src)}`,
    );
  }

  private oneType(ctor: string): TypeExpr {
    return this.args(ctor, 1)[0]!;
  }

  /** Read comma-separated types up to `)`: exactly `count`, or one or more. */
  private args(ctor: string, count: number | null): TypeExpr[] {
    if (this.peek() === ")") this.malformed(`${ctor}(...) needs a type`);
    const args = [this.type()];
    while (this.peek() === ",") {
      this.next();
      this.trailingComma(ctor);
      args.push(this.type());
    }
    this.expect(")");
    if (count !== null && args.length !== count) {
      const wanted = count === 1 ? "one type" : `${count} types`;
      this.malformed(`${ctor}(...) takes ${wanted}`);
    }
    return args;
  }

  /** Read `lo..hi)` after `int(` or `float(`: an inclusive range; either end may be open. */
  private range(ctor: "int" | "float"): ScalarExpr {
    let example = ctor === "int" ? "int(1..12)" : "float(0.0..1.0)";
    const low = this.endpoint();
    if (this.peek() === ",") {
      this.next();
      const high = this.endpoint();
      if ((low !== null || high !== null) && validBounds(ctor, [low, high])) {
        example = `${ctor}(${boundText(low)}..${boundText(high)})`;
      }
      this.malformed(`write a range: ${example}`);
    }
    const dots = this.peek() === "..";
    let high: Bound = null;
    if (dots) {
      this.next();
      high = this.endpoint();
    }
    const after = this.peek();
    if (after === null) this.ended();
    if (after === "," || after === "..") {
      this.malformed(`a range takes one lo..hi, e.g. ${example}`);
    }
    if (after !== ")") {
      throw new TypeCheckError(
        `numeric bounds require literal endpoints in ${reprText(this.src)}`,
      );
    }
    if (!dots) this.malformed(rangeNeedsDots(ctor));
    if (low === null && high === null) {
      this.malformed(`a range needs at least one end; write ${ctor} for any ${ctor}`);
    }
    this.expect(")");
    return rangeType(ctor, [low, high], this.src);
  }

  /** Read a numeric endpoint, or return `null` when the next token is not a number. */
  private endpoint(): Bound {
    if (this.peekKind() !== "number") return null;
    const text = this.next().text;
    if (/^[+-]?[0-9]+$/.test(text)) return BigInt(text);
    // `-0.0` is stored as `0`, so a -0 end never reaches a manifest or fingerprint.
    const value = Number(text);
    return value === 0 ? 0 : value;
  }

  private literalTags(): string[] {
    const tags: string[] = [];
    for (;;) {
      const { kind, text } = this.next();
      if (kind === "string") {
        tags.push(literalString(text, this.src));
      } else if (kind === "name" && !BUILTIN_CONSTANTS.has(text) && this.peek() !== "(") {
        tags.push(text);
      } else if (text === ")" && tags.length === 0) {
        this.malformed("Literal(...) needs a member");
      } else {
        throw new TypeCheckError(
          `invalid Literal member in ${reprText(this.src)} (use names or quoted strings)`,
        );
      }
      if (this.peek() === ",") {
        this.next();
        this.trailingComma("Literal");
        continue;
      }
      this.expect(")");
      return tags;
    }
  }
}

/** Spell the hint for a numeric range written without `..`. */
function rangeNeedsDots(ctor: "int" | "float"): string {
  const examples =
    ctor === "int"
      ? "int(1..12), int(0..), or int(..100)"
      : "float(0.0..1.0), float(0.0..), or float(..1.0)";
  return `a numeric range needs \`..\`, e.g. ${examples}`;
}

/** The error for type text `src` nested deeper than `limit` levels. */
function nestedTooDeeply(src: string, limit: number): TypeCheckError {
  const head = Array.from(src).slice(0, 40).join("");
  return new TypeCheckError(
    `type nested too deeply in ${reprText(head)}... (at most ${limit} levels)`,
  );
}

/** Build the range type `ctor(bounds)`, rejecting invalid bounds in type text `src`. */
function rangeType(ctor: "int" | "float", bounds: Bounds, src: string): ScalarExpr {
  try {
    validateNumericBounds(SCALAR_KIND.get(ctor)!, bounds);
  } catch (error) {
    if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
    throw new TypeCheckError(`invalid bounds in ${reprText(src)}: ${error.message}`, {
      cause: error,
    });
  }
  return { kind: "scalar", name: ctor, bounds };
}

/** Return whether `bounds` make a valid range for `ctor`. */
function validBounds(ctor: "int" | "float", bounds: Bounds): boolean {
  try {
    validateNumericBounds(SCALAR_KIND.get(ctor)!, bounds);
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) return false;
    throw error;
  }
  return true;
}

/** What each typing constructor takes and a sample argument list, for a name written bare. */
const CONSTRUCTOR_ARGS: ReadonlyMap<string, readonly [takes: string, example: string]> = new Map([
  ["list", ["a type", "X"]],
  ["List", ["a type", "X"]],
  ["Optional", ["a type", "X"]],
  ["Inbox", ["a type", "X"]],
  ["dict", ["two types", "K, V"]],
  ["Dict", ["two types", "K, V"]],
  ["Union", ["records", "A, B"]],
  ["Literal", ["members", "a, b"]],
]);

/**
 * Return the hint for type text `text` that is a typing constructor's bare name, else `null`.
 *
 * Example: `bareConstructorHint("list")` gives `"'list' takes a type: write list(X)"`.
 */
export function bareConstructorHint(text: string): string | null {
  const usage = CONSTRUCTOR_ARGS.get(text);
  if (usage === undefined) return null;
  return `${reprText(text)} takes ${usage[0]}: write ${text}(${usage[1]})`;
}

/**
 * Read a bare type name: a scalar keyword, a built-in memory type, or a registry reference.
 *
 * @throws TypeCheckError for the lowercase `any`, the retired `topics`, or a typing constructor
 *   written without its arguments (each hint shows the spelling to write).
 */
function namedType(name: string): TypeExpr {
  const hint = bareConstructorHint(name);
  if (hint !== null) throw new TypeCheckError(hint);
  if (name === "any") {
    // `any` stays a builtin name (the expression function), so it never reads as a
    // registry reference.
    throw new TypeCheckError("unknown type 'any'; write Any");
  }
  if (name === "topics") throw new TypeCheckError("unknown type 'topics'; write list(str)");
  // The historical lowercase token.
  if (name === "none") return { kind: "scalar", name: "None" };
  if (SCALAR_KEYWORDS.has(name)) return { kind: "scalar", name: name as ScalarName };
  if (BUILTIN_MEMORY_TYPES.has(name)) return { kind: "memory", type: name, item: null };
  return { kind: "ref", name };
}

/** Decode one quoted `Literal` member and check it is a usable tag. */
function literalString(text: string, src: string): string {
  const value = decodeQuoted(text, src);
  checkLiteralTag(value, src);
  return value;
}

const ESCAPES: ReadonlyMap<string, string> = new Map([
  ['"', '"'],
  ["'", "'"],
  ["\\", "\\"],
  ["/", "/"],
  ["b", "\b"],
  ["f", "\f"],
  ["n", "\n"],
  ["r", "\r"],
  ["t", "\t"],
]);

/**
 * Decode a quoted `Literal` member token (either quote style) with JSON's escapes.
 *
 * A high-low `\uXXXX` pair decodes to one character; a lone surrogate is left for
 * `checkLiteralTag` to refuse. Raw characters other than LF and CR are kept.
 */
function decodeQuoted(text: string, src: string): string {
  const fail = (detail: string): never => {
    throw malformedError(src, detail);
  };
  const body = text.slice(1, -1);
  if (/[\n\r]/.test(body)) fail("a quoted Literal member can't hold a line break");
  let out = "";
  let i = 0;
  while (i < body.length) {
    const ch = body[i]!;
    if (ch !== "\\") {
      out += ch;
      i += 1;
      continue;
    }
    // The tokenizer puts a character after every backslash.
    const escape = String.fromCodePoint(body.codePointAt(i + 1)!);
    if (escape === "u") {
      const hex = body.slice(i + 2, i + 6);
      if (!/^[0-9A-Fa-f]{4}$/.test(hex)) {
        fail("'\\u' needs four hex digits in a quoted Literal member");
      }
      out += String.fromCharCode(parseInt(hex, 16));
      i += 6;
      continue;
    }
    const decoded = ESCAPES.get(escape);
    if (decoded === undefined) {
      fail(`invalid escape '\\${escape}' in a quoted Literal member (use JSON escapes)`);
    }
    out += decoded;
    i += 1 + escape.length;
  }
  return out;
}

/** Reject an empty quoted `Literal` member or one holding a Unicode surrogate. */
function checkLiteralTag(value: string, src: string): void {
  if (!value) {
    throw new TypeCheckError(
      `empty Literal member in ${reprText(src)} ` +
        "(tags must be non-empty names or quoted strings)",
    );
  }
  if (!encodesAsUtf8(value)) {
    // An escape such as "\uD800" in the quoted member decodes to a lone surrogate.
    throw new TypeCheckError(
      `Literal member ${reprText(value)} in ${reprText(src)} contains a Unicode surrogate ` +
        "(tags must be valid UTF-8 text)",
    );
  }
}

// Decimal literals (`1.` and `.5` are floats) and identifiers. Signs are separate
// tokens, since the bracket spelling reads them as unary operators.
const LEGACY_TOKEN = new RegExp(
  String.raw`\s*(?:(?<number>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)` +
    `|(?<name>${IDENTIFIER_PATTERN})|(?<string>${QUOTED})` +
    String.raw`|(?<punct>[[\],|+-]))`,
  "uy",
);

/** A node of bracket-spelled type text. */
type LegacyNode =
  | { readonly kind: "name"; readonly id: string }
  | { readonly kind: "constant"; readonly value: string }
  | { readonly kind: "number"; readonly value: bigint | number }
  | { readonly kind: "string"; readonly value: string }
  | { readonly kind: "signed"; readonly signs: string; readonly operand: LegacyNode }
  | { readonly kind: "subscript"; readonly base: LegacyNode; readonly slice: LegacyNode }
  | { readonly kind: "tuple"; readonly items: readonly LegacyNode[] }
  | { readonly kind: "list"; readonly items: readonly LegacyNode[] }
  | { readonly kind: "bar"; readonly operands: readonly LegacyNode[] };

/**
 * Parse the older bracket spelling (`Optional[str]`, `int[1, 12]`).
 *
 * Only migration errors use it, to show the parenthesized spelling.
 */
function parseLegacyType(text: string): TypeExpr {
  const src = text.trim();
  checkEncodable(src);
  return typeFromLegacy(legacySyntax(src), src);
}

/**
 * Read bracket-spelled type text into a `LegacyNode` tree: names, numbers, quoted strings
 * (JSON escapes), `None`/`True`/`False`, unary signs, subscripts, list displays and `|`.
 */
function legacySyntax(src: string): LegacyNode {
  // Closures rather than a class: the grammar is a handful of rules over shared state,
  // used once per call.
  const malformed = (): never => {
    throw malformedError(src);
  };
  const tokens = tokenize(LEGACY_TOKEN, src) ?? malformed();
  let index = 0;
  let depth = 0;
  const maxDepth = current().maxTypeNesting;
  const peek = (): string | null => tokens[index]?.text ?? null;
  const next = (): Token => tokens[index++] ?? malformed();

  // Read `item, item, ...` up to `]` after a `[`; `comma` tells whether one was written.
  const bracketed = (): { items: LegacyNode[]; comma: boolean } => {
    depth += 1;
    if (depth > maxDepth) throw nestedTooDeeply(src, maxDepth);
    const items: LegacyNode[] = [];
    let comma = false;
    while (peek() !== "]") {
      items.push(expr());
      if (peek() !== ",") break;
      index += 1;
      comma = true;
    }
    if (next().text !== "]") malformed();
    depth -= 1;
    return { items, comma };
  };
  const atom = (): LegacyNode => {
    const { kind, text } = next();
    if (kind === "name") {
      // Keywords match before identifiers are read in NFKC form, so `ｓｔｒ` is
      // `str` but `Ｔｒｕｅ` is a name, not the constant.
      if (BUILTIN_CONSTANTS.has(text)) return { kind: "constant", value: text };
      return { kind, id: text.normalize("NFKC") };
    }
    if (kind === "string") return { kind, value: decodeQuoted(text, src) };
    if (kind === "number") {
      if (!/^[0-9]+$/.test(text)) return { kind, value: Number(text) };
      // A decimal int other than zero itself can't have a leading zero.
      if (/^0+[1-9]/.test(text)) malformed();
      return { kind, value: BigInt(text) };
    }
    if (text !== "[") malformed();
    return { kind: "list", items: bracketed().items };
  };
  const operand = (): LegacyNode => {
    let signs = "";
    while (peek() === "+" || peek() === "-") signs += next().text;
    let node = atom();
    while (peek() === "[") {
      index += 1;
      const { items, comma } = bracketed();
      if (items.length === 0) malformed();
      node = { kind: "subscript", base: node, slice: comma ? { kind: "tuple", items } : items[0]! };
    }
    return signs ? { kind: "signed", signs, operand: node } : node;
  };
  const expr = (): LegacyNode => {
    const operands = [operand()];
    while (peek() === "|") {
      index += 1;
      operands.push(operand());
    }
    return operands.length === 1 ? operands[0]! : { kind: "bar", operands };
  };

  const node = expr();
  if (index < tokens.length) malformed();
  return node;
}

/** Convert one node of bracket-spelled type text into a `TypeExpr`. */
function typeFromLegacy(node: LegacyNode, src: string): TypeExpr {
  if (node.kind === "constant" && node.value === "None") return { kind: "scalar", name: "None" };
  if (node.kind === "name") return namedType(node.id);
  if (node.kind === "subscript") {
    if (node.base.kind !== "name") throw malformedError(src);
    const ctor = node.base.id;
    const sl = node.slice;
    const elts = sl.kind === "tuple" ? sl.items : [sl];
    switch (ctor) {
      case "int":
      case "float": {
        if (sl.kind !== "tuple" || sl.items.length !== 2) {
          throw malformedError(src, rangeNeedsDots(ctor));
        }
        const low = numericEndpoint(sl.items[0]!, src);
        const high = numericEndpoint(sl.items[1]!, src);
        return rangeType(ctor, [low, high], src);
      }
      case "list":
      case "List":
        return { kind: "list", element: typeFromLegacy(sl, src) };
      case "Optional":
        return { kind: "optional", inner: typeFromLegacy(sl, src) };
      case "Literal":
        return { kind: "enum", tags: literalTags(elts, src) };
      case "dict":
      case "Dict":
        return { kind: "dict" };
      case "Union": {
        const members = elts.map((e) => typeFromLegacy(e, src));
        if (members.some((m) => m.kind === "scalar" && m.name === "None")) {
          throw new TypeCheckError(
            `None is not a union member in ${reprText(src)}; use Optional(Union(...))`,
          );
        }
        return { kind: "union", members };
      }
    }
    throw new TypeCheckError(`unknown type constructor ${reprText(ctor)} in ${reprText(src)}`);
  }
  if (node.kind === "bar") {
    const operands = node.operands.map((operand) => legacyLabel(operand, src)).join(", ");
    throw malformedError(src, `use Union(${operands}) (| is not a type operator)`);
  }
  throw malformedError(src);
}

/** Spell one `A | B` operand in the parenthesized syntax, or as written if it is no type. */
function legacyLabel(node: LegacyNode, src: string): string {
  try {
    return formatTypeExpr(typeFromLegacy(node, src));
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    return unparse(node);
  }
}

/** Spell a node as written. */
function unparse(node: LegacyNode): string {
  switch (node.kind) {
    case "name":
      return node.id;
    case "constant":
      return node.value;
    case "number":
      return boundText(node.value);
    case "string":
      return reprText(node.value);
    case "signed":
      return node.signs + unparse(node.operand);
    case "subscript":
      return `${unparse(node.base)}[${unparse(node.slice)}]`;
    case "tuple":
      return node.items.map(unparse).join(", ") + (node.items.length === 1 ? "," : "");
    case "list":
      return `[${node.items.map(unparse).join(", ")}]`;
    case "bar":
      return node.operands.map(unparse).join(" | ");
  }
}

/** Read one numeric literal with at most one unary sign, without evaluation. */
function numericEndpoint(node: LegacyNode, src: string): bigint | number {
  let negative = false;
  if (node.kind === "signed" && node.signs.length === 1) {
    negative = node.signs === "-";
    node = node.operand;
  }
  if (node.kind !== "number") {
    throw new TypeCheckError(`numeric bounds require literal endpoints in ${reprText(src)}`);
  }
  if (!negative) return node.value;
  // `-0.0` is stored as `0`, so a -0 end never reaches a manifest or fingerprint.
  const value = -node.value;
  return value === 0 ? 0 : value;
}

/** Read the members of a bracket-spelled `Literal[...]`. */
function literalTags(elts: readonly LegacyNode[], src: string): string[] {
  return elts.map((e) => {
    if (e.kind === "string") {
      checkLiteralTag(e.value, src);
      return e.value;
    }
    if (e.kind === "name") return e.id;
    throw new TypeCheckError(
      `invalid Literal member in ${reprText(src)} (use names or quoted strings)`,
    );
  });
}

// --------------------------------------------------------------------------------------- //
// Registry and resolution
// --------------------------------------------------------------------------------------- //

/** A record typedef: each field name's type text, in authored order. */
export interface RecordDef {
  readonly kind: "record";
  readonly fields: ReadonlyMap<string, string>;
}

/** A `Literal(...)` typedef: its tags. */
export interface VariantDef {
  readonly kind: "variant";
  readonly tags: readonly string[];
}

/** A transparent type abbreviation, resolved transitively (cycle-guarded). */
export interface AliasDef {
  readonly kind: "alias";
  readonly target: TypeExpr;
}

/** A host's own memory type, named by its class's contract; `withMemoryTypes` adds it. */
export interface MemoryDef {
  readonly kind: "memory";
}

/** One typedef of a flow, or a host memory type. */
export type TypeDef = RecordDef | VariantDef | AliasDef | MemoryDef;

/** A flow's typedefs by name, which a `RefExpr` resolves against. */
export type TypeRegistry = ReadonlyMap<string, TypeDef>;

/** A record typedef from its field type texts. */
function record(fields: Readonly<Record<string, string>>): RecordDef {
  return { kind: "record", fields: new Map(Object.entries(fields)) };
}

// The built-in records, in Yian's own type syntax. Only `Message` is public: the parts are reached
// through it, so their names stay free for typedefs.
const BUILTIN_RECORDS: TypeRegistry = new Map<string, TypeDef>([
  ["Message", record({
    role: "Literal(user, assistant, tool)",
    parts: "list(Union(Text, Reasoning, Refusal, ToolCall, ToolResult, Media))",
    author: "Optional(str)",
    protocol: "Optional(str)",
    provider_payload: "Optional(object)",
  })],
  ["Text", record({ type: "Literal(text)", text: "str", provider_payload: "Optional(object)" })],
  ["Reasoning", record({ type: "Literal(reasoning)", text: "str", provider_payload: "Optional(object)" })],
  ["Refusal", record({ type: "Literal(refusal)", text: "str", provider_payload: "Optional(object)" })],
  ["ToolCall", record({
    type: "Literal(tool_call)", id: "str", name: "str", arguments: "Optional(object)",
    raw_arguments: "Optional(str)", provider_payload: "Optional(object)",
  })],
  ["ToolResult", record({
    type: "Literal(tool_result)", call_id: "str", name: "str", content: "str",
    is_error: "Optional(bool)", provider_payload: "Optional(object)",
  })],
  ["Media", record({
    type: "Literal(media)", mime: "str", file: "file", name: "Optional(str)", provider_payload: "Optional(object)",
  })],
]);

const builtinTypes = new Map<string, Type>();

/**
 * Return the resolved built-in record type `name` (`Message`), or `null` for any other name. A
 * flow's registry never holds these names (`readTypedefs` refuses them), so `resolveType` asks
 * here only after the registry.
 */
export function builtinType(name: string): Type | null {
  if (!BUILTIN_RECORD_NAMES.has(name)) return null;
  let typ = builtinTypes.get(name);
  if (typ === undefined) {
    typ = resolveType({ kind: "ref", name }, BUILTIN_RECORDS);
    builtinTypes.set(name, typ);
  }
  return typ;
}

let messagesType: Type | undefined;

/** Return the type of an AGENT's new messages, `list(Message)`. */
export function newMessagesType(): Type {
  messagesType ??= typeFor("list(Message)", new Map());
  return messagesType;
}

/**
 * Resolve a `TypeExpr` against the registry into a runtime `Type`.
 *
 * Expands records field by field and follows aliases transitively, guarding against
 * reference cycles via `seen`. A `Union(...)` flattens nested unions and aliases, drops
 * duplicate members, and requires every member to be a non-nullable record. Its members
 * must be told apart by a tag field or by shapes that can't overlap. A union with one
 * distinct member is that member. The result is a scalar, a typed list, a closed record,
 * a tag-constrained variant, a union of records with members in canonical order, an inbox or a
 * memory. An inbox is a whole input: never optional, never a list item, record field or union
 * member, and never holding an inbox. A memory is a whole input too, but may be optional; a
 * `VectorDBMemory(X)`'s item `X` must be a record.
 *
 * `seen` holds the registry names already being resolved on the current path, the cycle
 * guard; callers pass nothing.
 *
 * Example: with `Buy: {action: Literal(buy), qty: int}` and
 * `Sell: {action: Literal(sell), price: float}` registered,
 * `resolveType(parseType("Union(Sell, Buy)"), registry)` gives a union whose members are
 * `Buy` then `Sell`.
 *
 * @throws TypeCheckError when a `RefExpr` names an unknown type, a record or alias
 *   reference is recursive, a union member is not a record or is nullable, two union
 *   members can match the same value, or an inbox or a memory sits where it can't.
 */
export function resolveType(
  t: TypeExpr,
  registry: TypeRegistry,
  seen: ReadonlySet<string> = new Set(),
): Type {
  switch (t.kind) {
    case "scalar": {
      const kind = SCALAR_KIND.get(t.name)!;
      if (t.bounds !== undefined) validateNumericBounds(kind, t.bounds);
      return new Type({ kind, bounds: t.bounds });
    }
    case "optional": {
      // Peel the whole chain in a loop so nested Optionals cost one frame, not one each.
      let inner: TypeExpr = t;
      while (inner.kind === "optional") inner = inner.inner;
      const resolved = resolveType(inner, registry, seen);
      if (resolved.kind === ValueKind.INBOX) {
        throw new TypeCheckError(`an inbox is never optional in ${reprText(formatTypeExpr(t))}`);
      }
      return resolved.replace({ nullable: true });
    }
    case "enum":
      return new Type({ kind: ValueKind.STRING, tags: new Set(t.tags) });
    case "dict": {
      const wholes = (t.args ?? []).map((arg) => namesWhole(arg, registry, seen));
      if (wholes.includes("inbox")) {
        throw new TypeCheckError(
          `an inbox is a whole input; it can't be a dict key or value in ${reprText(formatTypeExpr(t))}`,
        );
      }
      if (wholes.includes("memory")) {
        throw new TypeCheckError(`a memory is never a dict key or value in ${reprText(formatTypeExpr(t))}`);
      }
      // Only names are checked: the dict stays a lenient object, so a record may hold itself
      // through one and its members need not resolve as a type of their own.
      for (const arg of t.args ?? []) checkLocal(arg, registry);
      return Type.scalar(ValueKind.OBJECT); // lenient (no key/value typing yet)
    }
    case "union":
      return resolveUnion(t, registry, seen);
    case "list": {
      const element = resolveType(t.element, registry, seen);
      if (element.kind === ValueKind.INBOX) {
        throw new TypeCheckError(
          `an inbox is a whole input; it can't be a list item in ${reprText(formatTypeExpr(t))}`,
        );
      }
      if (isMemoryType(element)) {
        throw new TypeCheckError(`a memory is never a list item in ${reprText(formatTypeExpr(t))}`);
      }
      return new Type({ kind: listKindForElement(element), element });
    }
    case "inbox": {
      const element = resolveType(t.element, registry, seen);
      if (element.kind === ValueKind.INBOX) {
        throw new TypeCheckError(
          `an inbox holds values, not inboxes, in ${reprText(formatTypeExpr(t))}`,
        );
      }
      if (isMemoryType(element)) {
        throw new TypeCheckError(`an inbox holds values, not memories, in ${reprText(formatTypeExpr(t))}`);
      }
      return inboxType(element);
    }
    case "memory": {
      if (t.item === null) return memoryType(t.type, null);
      // The item is named by one of the flow's own record typedefs, whose fields type each item.
      const named = t.item.kind === "ref" && registry.has(t.item.name);
      const item = named ? resolveType(t.item, registry, seen) : null;
      if (item === null || item.fields === undefined || item.nullable) {
        throw new TypeCheckError(
          `${t.type}'s item type must be a record typedef in ${reprText(formatTypeExpr(t))}`,
        );
      }
      return memoryType(t.type, item);
    }
    case "ref":
      break;
  }
  const name = t.name;
  if (seen.has(name)) throw new TypeCheckError(`recursive type reference: ${reprText(name)}`);
  const defn = registry.get(name);
  if (defn === undefined) {
    const builtin = builtinType(name);
    if (builtin !== null) return builtin;
    throw new TypeCheckError(`unknown type ${reprText(name)}`);
  }
  if (defn.kind === "memory") return memoryType(name, null);
  const inner = new Set([...seen, name]);
  if (defn.kind === "alias") return resolveType(defn.target, registry, inner);
  if (defn.kind === "variant") {
    return new Type({ kind: ValueKind.STRING, tags: new Set(defn.tags) });
  }
  const fields = new Map<string, Type>();
  for (const [field, text] of defn.fields) {
    const typ = resolveType(parseType(text), registry, inner);
    rejectWholeField(typ);
    fields.set(field, typ);
  }
  // An `Optional(X)` field (its resolved type is nullable) is not required.
  const required = new Set([...fields].filter(([, typ]) => !typ.nullable).map(([f]) => f));
  return new Type({ kind: ValueKind.OBJECT, fields, required });
}

/** Refuse `typ` as a record field's type when it is an inbox or a memory. */
function rejectWholeField(typ: Type): void {
  if (typ.kind === ValueKind.INBOX) {
    throw new TypeCheckError("an inbox is a whole input; it can't be a record field");
  }
  if (isMemoryType(typ)) throw new TypeCheckError("a memory is never a record field");
}

/** One flattened union member: its display name, its origin and its resolved type. */
type UnionPart = readonly [name: string, origin: string | null, member: Type];

/** Resolve a `Union(...)` into a record, or a union of distinguishable records. */
function resolveUnion(t: UnionExpr, registry: TypeRegistry, seen: ReadonlySet<string>): Type {
  // The label is rendered only for an error: rendering recurses through the whole
  // expression, which a deeply nested union would pay at every level of resolution.
  const members: UnionPart[] = [];
  const parts = unionParts(t, registry, seen);
  // Checked before the record checks, so a memory member is named as one wherever it stands.
  if (parts.some(([, , member]) => isMemoryType(member))) {
    throw new TypeCheckError(`a memory is never a union member in ${reprText(formatTypeExpr(t))}`);
  }
  for (const [name, origin, member] of parts) {
    if (member.nullable) {
      throw new TypeCheckError(
        `${formatTypeExpr(t)}: member ${reprText(name)} is nullable; use ` +
          "Optional(Union(...)) instead",
      );
    }
    if (member.fields === undefined) {
      throw new TypeCheckError(`${formatTypeExpr(t)}: member ${reprText(name)} is not a record`);
    }
    const same = members.find(([, , m]) => typeEquals(m, member));
    if (same === undefined) {
      members.push([name, origin, member]);
    } else if (same[1] !== origin) {
      // Two differently named records with one shape can't be told apart.
      throw overlapError(formatTypeExpr(t), same[0], name);
    }
  }
  if (members.length === 1) return members[0]![2];
  const union = new Type({
    kind: ValueKind.OBJECT,
    members: canonicalUnionOrder(members.map(([, , member]) => member)),
  });
  if (unionTagFields(union).length === 0) {
    for (const [i, [aName, , a]] of members.entries()) {
      for (const [bName, , b] of members.slice(i + 1)) {
        if (typesOverlap(a, b)) {
          throw overlapError(formatTypeExpr(t), aName, bName, holdsFile(a) || holdsFile(b));
        }
      }
    }
  }
  return union;
}

/**
 * Whether `t` is `file`, or a record or union with a field typed `file` at any record
 * depth. List elements are not walked, since any two lists already overlap.
 */
function holdsFile(t: Type): boolean {
  if (t.kind === ValueKind.FILE) return true;
  if (t.fields !== undefined) return [...t.fields.values()].some(holdsFile);
  return t.members !== undefined && t.members.some(holdsFile);
}

/**
 * Build the error for two union members, `aName` and `bName`, that can match the same
 * value in the union `label` as written. `files` says whether either member holds a
 * `file` field, which overlaps more than its name suggests, so the message says why.
 */
function overlapError(label: string, aName: string, bName: string, files = false): TypeCheckError {
  const why = files
    ? " (a `file` slot reads `{uri, mime, name}`, so `file` overlaps `object`, `Any` " +
      "and a record of that shape)"
    : "";
  return new TypeCheckError(
    `${label}: members ${reprText(aName)} and ${reprText(bName)} can match the same ` +
      `value${why}; add a tag field every member declares as a single-value Literal, with ` +
      `a different value in each member, e.g. kind: ${literalLabel([aName.toLowerCase()])}`,
  );
}

/**
 * Return a union's flattened `[display name, origin, resolved member]` triples.
 *
 * Nested unions and aliases are expanded so error messages can name each member the way
 * it was written: a member reached through an alias chain is named by the chain's first
 * alias, unless a union lies between. The origin is the registry name an alias chain ends
 * at, or `null` when the member is not a reference; it tells a repeated member from a
 * differently named record of the same shape.
 */
function unionParts(t: TypeExpr, registry: TypeRegistry, seen: ReadonlySet<string>): UnionPart[] {
  // An explicit stack, so flattening deeply nested unions and long alias chains costs no
  // frames. Each entry is [expression, names on its alias path, display name so far].
  const parts: UnionPart[] = [];
  const stack: [TypeExpr, ReadonlySet<string>, string | null][] = [[t, seen, null]];
  while (stack.length > 0) {
    const [expr, path, name] = stack.pop()!;
    if (expr.kind === "union") {
      for (const member of expr.members.toReversed()) stack.push([member, path, null]);
      continue;
    }
    if (expr.kind === "ref") {
      if (path.has(expr.name)) {
        throw new TypeCheckError(`recursive type reference: ${reprText(expr.name)}`);
      }
      const defn = registry.get(expr.name);
      if (defn?.kind === "alias") {
        stack.push([defn.target, new Set([...path, expr.name]), name ?? expr.name]);
        continue;
      }
    }
    const origin = expr.kind === "ref" ? expr.name : null;
    parts.push([name ?? formatTypeExpr(expr), origin, resolveType(expr, registry, path)]);
  }
  return parts;
}

/**
 * Render a parsed type in the parenthesized spelling `parseType` reads back.
 *
 * A float range end prints in its shortest form with a fraction digit or exponent (`1.0`,
 * `1e-05`) and an int end as its digits, as labels spell them. A dict read without its key
 * and value types (from the bracket spelling) renders as `object`, the lenient object it reads as.
 *
 * Example: `formatTypeExpr(parseType("Optional(Union(Buy,Sell))"))` gives
 * `"Optional(Union(Buy, Sell))"`.
 */
export function formatTypeExpr(t: TypeExpr): string {
  switch (t.kind) {
    case "scalar":
      if (t.bounds === undefined) return t.name;
      return `${t.name}(${boundText(t.bounds[0])}..${boundText(t.bounds[1])})`;
    case "ref":
      return t.name;
    case "list":
      return `list(${formatTypeExpr(t.element)})`;
    case "inbox":
      return `Inbox(${formatTypeExpr(t.element)})`;
    case "memory":
      return t.item === null ? t.type : `${t.type}(${formatTypeExpr(t.item)})`;
    case "optional":
      return `Optional(${formatTypeExpr(t.inner)})`;
    case "enum":
      return literalLabel(t.tags);
    case "union":
      return "Union(" + t.members.map(formatTypeExpr).join(", ") + ")";
    case "dict":
      if (t.args === undefined) return "object";
      return "dict(" + t.args.map(formatTypeExpr).join(", ") + ")";
  }
}

/**
 * Parse type text and resolve it against the registry in one step.
 *
 * @throws TypeCheckError when `text` does not parse or resolve, or the resolved type nests
 *   more record and list levels than the type nesting limit: text within the limit can
 *   still name a typedef that brings it past.
 */
export function typeFor(text: string, registry: TypeRegistry): Type {
  const typ = resolveType(parseType(text), registry);
  const limit = current().maxTypeNesting;
  if (typeLevels(typ) > limit) {
    throw new TypeCheckError(
      `type nests deeper than ${limit} levels once its typedef names are expanded, ` +
        "the limit for a type (each record and list is a level)",
    );
  }
  return typ;
}

// --------------------------------------------------------------------------------------- //
// Typedef registry builder
// --------------------------------------------------------------------------------------- //

/** A typedef's parsed `[field, type]` parts: each record field, or the alias target. */
type ParsedParts = readonly (readonly [field: string | null, type: TypeExpr])[];

/** The registry names a parsed type references (for the eager alias-cycle check). */
function refsInType(t: TypeExpr): Set<string> {
  switch (t.kind) {
    case "ref":
      return new Set([t.name]);
    case "list":
    case "inbox":
      return refsInType(t.element);
    case "memory":
      return t.item === null ? new Set() : refsInType(t.item);
    case "optional":
      return refsInType(t.inner);
    case "union":
      return new Set(t.members.flatMap((member) => [...refsInType(member)]));
    default:
      return new Set(); // scalars, enums and the dict placeholder reference nothing
  }
}

/**
 * Return `"inbox"` or `"memory"` when type expression `t` is or holds an inbox or a memory (the
 * first found), following alias names in `registry`, else `null`. A name that is unknown, a
 * record, or already on the `seen` path holds neither here.
 */
function namesWhole(
  t: TypeExpr,
  registry: TypeRegistry,
  seen: ReadonlySet<string>,
): "inbox" | "memory" | null {
  const first = (parts: readonly TypeExpr[]) => {
    for (const part of parts) {
      const whole = namesWhole(part, registry, seen);
      if (whole !== null) return whole;
    }
    return null;
  };
  switch (t.kind) {
    case "inbox":
      return "inbox";
    case "memory":
      return "memory";
    case "list":
      return namesWhole(t.element, registry, seen);
    case "optional":
      return namesWhole(t.inner, registry, seen);
    case "union":
      return first(t.members);
    case "dict":
      return first(t.args ?? []);
    case "ref": {
      const defn = registry.get(t.name);
      if (defn?.kind === "memory") return "memory";
      if (defn?.kind !== "alias" || seen.has(t.name)) return null;
      return namesWhole(defn.target, registry, new Set([...seen, t.name]));
    }
    default:
      return null;
  }
}

/** Return a typedef's `[field, type]` parts: each record field, or the alias target. */
function typedefParts(defn: TypeDef): [string | null, string | TypeExpr][] {
  if (defn.kind === "record") return [...defn.fields];
  if (defn.kind === "alias") return [[null, defn.target]];
  return []; // an enum's tags are already checked by parseType
}

/**
 * Prefix error `exc` with the typedef `name` (and `field`) it belongs to, and locate it
 * there; `options` (such as a `cause`) pass through to the new error.
 */
function typedefError(
  name: string,
  field: string | null,
  exc: TypeCheckError,
  options: ErrorOptions = {},
): TypeCheckError {
  if (field === null) {
    return new TypeCheckError(`typedef ${reprText(name)}: ${exc.message}`, {
      ...options,
      path: [name],
    });
  }
  const where = `typedef ${reprText(name)} field ${reprText(field)}`;
  return new TypeCheckError(`${where}: ${exc.message}`, { ...options, path: [name, field] });
}

/** Check what a type expression says on its own: names it uses exist, bounds are valid. */
function checkLocal(t: TypeExpr, registry: TypeRegistry): void {
  switch (t.kind) {
    case "scalar":
      if (t.bounds !== undefined) validateNumericBounds(SCALAR_KIND.get(t.name)!, t.bounds);
      break;
    case "ref":
      if (!registry.has(t.name) && builtinType(t.name) === null) {
        throw new TypeCheckError(`unknown type ${reprText(t.name)}`);
      }
      break;
    case "list":
    case "inbox":
      checkLocal(t.element, registry);
      break;
    case "memory":
      if (t.item !== null) checkLocal(t.item, registry);
      break;
    case "optional":
      checkLocal(t.inner, registry);
      break;
    case "union":
      for (const member of t.members) checkLocal(member, registry);
      break;
    case "dict":
      for (const arg of t.args ?? []) checkLocal(arg, registry);
      break;
    default:
      break; // enums have nothing to check locally
  }
}

/**
 * Return how many record and list levels a parsed type nests.
 *
 * Each `list(...)` is a level, `Optional(...)` adds none, a union counts as its deepest
 * member, and a name counts as `levels.get(name)` (0 when not yet measured, which happens
 * only on a reference cycle). Other scalars, `Literal(...)` and `dict(...)` are 0.
 */
function exprLevels(t: TypeExpr, levels: ReadonlyMap<string, number>): number {
  // Levels are the most lists on any path to a leaf plus that leaf's levels, so an
  // explicit stack of [expression, lists above it] needs no recursion.
  let deepest = 0;
  const stack: [TypeExpr, number][] = [[t, 0]];
  while (stack.length > 0) {
    const [expr, above] = stack.pop()!;
    if (expr.kind === "list") {
      stack.push([expr.element, above + 1]);
    } else if (expr.kind === "optional" || expr.kind === "inbox") {
      stack.push([expr.kind === "inbox" ? expr.element : expr.inner, above]);
    } else if (expr.kind === "memory" && expr.item !== null) {
      stack.push([expr.item, above]);
    } else if (expr.kind === "union") {
      for (const member of expr.members) stack.push([member, above]);
    } else {
      const leaf = expr.kind === "ref" ? (levels.get(expr.name) ?? 0) : 0;
      deepest = Math.max(deepest, above + leaf);
    }
  }
  return deepest;
}

/** The largest of `counts`, or 0 when there are none. */
function most(counts: readonly number[]): number {
  // A reduce, not `Math.max(...counts)`: a spread's argument count is capped.
  return counts.reduce((a, b) => Math.max(a, b), 0);
}

/**
 * Measure typedef `name` (definition `defn`, parsed `parts`) from the typedefs it names, and
 * store its levels and chain under `name` in `levels` and `chain`, where a name not yet
 * stored counts 0.
 *
 * Returns its levels (a record is 1 more than its deepest field, an alias its target's, an
 * enum 0; equal to `typeLevels` of the resolved type), its chain (the typedefs on the
 * longest path of names from it, itself included), the first field with the most levels,
 * and the first field with the longest chain. The fields are `null` for an alias or an enum.
 */
function measureTypedef(
  name: string,
  defn: TypeDef,
  parts: ParsedParts,
  levels: Map<string, number>,
  chain: Map<string, number>,
): [depth: number, length: number, levelsField: string | null, chainField: string | null] {
  const fieldLevels = parts.map(([field, t]) => [field, exprLevels(t, levels)] as const);
  const fieldChain = parts.map(
    ([field, t]) => [field, most(Array.from(refsInType(t), (ref) => chain.get(ref) ?? 0))] as const,
  );
  const deepest = most(fieldLevels.map(([, d]) => d));
  const longest = most(fieldChain.map(([, c]) => c));
  // An enum has no parts, so it measures 0 levels like the string it resolves to.
  const depth = (defn.kind === "record" ? 1 : 0) + deepest;
  const length = 1 + longest;
  levels.set(name, depth);
  chain.set(name, length);
  const levelsField = fieldLevels.find(([, d]) => d === deepest)?.[0] ?? null;
  const chainField = fieldChain.find(([, c]) => c === longest)?.[0] ?? null;
  return [depth, length, levelsField, chainField];
}

/**
 * Measure typedef `name` (definition `defn`, parsed `parts`) into the `levels` and `chain`
 * memos of `measureTypedef`, and reject it when its levels or chain exceed the type nesting
 * limit.
 *
 * @throws TypeCheckError when the typedef nests more record and list levels than the limit,
 *   or ends a longer chain of typedefs each naming the next. `path` is `[typedef, field]`
 *   for the record field reported, else `[typedef]`.
 */
function checkMeasure(
  name: string,
  defn: TypeDef,
  parts: ParsedParts,
  levels: Map<string, number>,
  chain: Map<string, number>,
): void {
  const limit = current().maxTypeNesting;
  const [depth, length, levelsField, chainField] = measureTypedef(
    name,
    defn,
    parts,
    levels,
    chain,
  );
  if (depth > limit) {
    let message =
      `type nests deeper than ${limit} levels, the limit for a type (each record and ` +
      "list is a level)";
    if (defn.kind === "record") {
      message += `; its type ${reprText(defn.fields.get(levelsField!)!)} adds ${depth - 1}`;
      throw typedefError(name, levelsField, new TypeCheckError(message));
    }
    throw typedefError(name, null, new TypeCheckError(message));
  }
  if (length > limit) {
    throw typedefError(
      name,
      chainField,
      new TypeCheckError(
        `it ends a chain of ${length} typedefs, each naming the next, more than ` +
          `${limit}, the limit for a type; write its type with fewer typedefs in between`,
      ),
    );
  }
}

/**
 * Resolve every typedef, so a broken one fails even when nothing uses it.
 *
 * Typedefs are checked dependencies first, each one's fields in order. Each typedef is first
 * measured: its levels and its chain of typedef names must stay within the type nesting
 * limit, so resolution's stack stays bounded by the limit. A cycle is reported at the field
 * that closes it, naming the typedef that holds that field: with `A: {x: B}` and
 * `B: {y: A}` the error is `typedef 'B' field 'y': recursive type reference: 'B'`.
 */
function checkTypedefs(registry: TypeRegistry): void {
  const parsed = new Map<string, ParsedParts>();
  // First each typedef's own text, so an error names the typedef that holds the mistake
  // rather than the first typedef that happens to use it.
  for (const [name, defn] of registry) {
    const parts: [string | null, TypeExpr][] = [];
    for (const [field, written] of typedefParts(defn)) {
      let t: TypeExpr;
      try {
        t = typeof written === "string" ? parseType(written) : written;
        checkLocal(t, registry);
      } catch (error) {
        if (!(error instanceof TypeCheckError)) throw error;
        throw typedefError(name, field, error, { cause: error });
      }
      parts.push([field, t]);
    }
    parsed.set(name, parts);
  }

  // Then the checks that follow references (recursion, union members), dependencies first
  // for the same reason. The walk is iterative so a long chain can't overflow the stack; a
  // ref still on the walk's stack (gray) when met is a back edge, so the typedef holding it
  // lies on a cycle.
  const order: string[] = [];
  const finished = new Set<string>();
  const backRefs = new Map<string, Set<string>>();

  const refsOf = (name: string): Iterator<string> =>
    parsed
      .get(name)!
      .flatMap(([, t]) => [...refsInType(t)].sort(compareCodeUnits))
      .values();

  // A typedef becomes a key of `backRefs` when the walk reaches it, so the keys are also
  // the set of typedefs already discovered.
  for (const root of registry.keys()) {
    if (backRefs.has(root)) continue;
    backRefs.set(root, new Set());
    const stack: [string, Iterator<string>][] = [[root, refsOf(root)]];
    while (stack.length > 0) {
      const [name, refs] = stack.at(-1)!;
      let descended = false;
      for (let step = refs.next(); !step.done; step = refs.next()) {
        const ref = step.value;
        // A built-in type is resolved, never walked.
        if (!registry.has(ref)) continue;
        if (!backRefs.has(ref)) {
          backRefs.set(ref, new Set());
          stack.push([ref, refsOf(ref)]);
          descended = true;
          break;
        }
        if (!finished.has(ref)) backRefs.get(name)!.add(ref);
      }
      if (!descended) {
        stack.pop();
        finished.add(name);
        order.push(name);
      }
    }
  }

  // A built-in counts its resolved levels and adds nothing to a chain.
  const levels = new Map<string, number>(
    [...BUILTIN_RECORD_NAMES].map((name) => [name, typeLevels(builtinType(name)!)]),
  );
  const chain = new Map<string, number>();
  for (const name of order) {
    const parts = parsed.get(name)!;
    checkMeasure(name, registry.get(name)!, parts, levels, chain);
    const back = backRefs.get(name)!;
    for (const [field, t] of parts) {
      if ([...refsInType(t)].some((ref) => back.has(ref))) {
        const exc = new TypeCheckError(`recursive type reference: ${reprText(name)}`);
        throw typedefError(name, field, exc);
      }
      let typ: Type;
      try {
        typ = resolveType(t, registry, new Set([name]));
        if (field !== null) rejectWholeField(typ);
      } catch (error) {
        if (!(error instanceof TypeCheckError)) throw error;
        throw typedefError(name, field, error, { cause: error });
      }
      // A memory is written only as a flow input's tag, so no name can build one from a value.
      if (isMemoryType(typ)) {
        throw new TypeCheckError(
          `typedef ${reprText(name)} names a memory type; declare memories as flow inputs (\`lessons: !FileMemory\`)`,
          { path: [name] },
        );
      }
    }
  }
}

/**
 * Reject an alias-to-alias cycle, located at the alias it was found from (record recursion
 * is caught by `checkTypedefs`).
 */
function checkAliasCycles(registry: TypeRegistry): void {
  // An iterative walk, so a long alias chain can't overflow the stack; an alias whose walk
  // finished without a cycle is not walked again, so a chain costs linear time.
  const finished = new Set<string>();
  const aliasRefs = (defn: AliasDef): Iterator<string> =>
    [...refsInType(defn.target)].sort(compareCodeUnits).values();

  for (const [start, defn] of registry) {
    if (defn.kind !== "alias" || finished.has(start)) continue;
    const onPath = new Set([start]);
    const stack: [string, Iterator<string>][] = [[start, aliasRefs(defn)]];
    while (stack.length > 0) {
      const [name, refs] = stack.at(-1)!;
      let descended = false;
      for (let step = refs.next(); !step.done; step = refs.next()) {
        const ref = step.value;
        if (onPath.has(ref)) {
          throw new TypeCheckError(`alias cycle through ${reprText(ref)} in typedefs`, {
            path: [start],
          });
        }
        const target = registry.get(ref);
        if (target?.kind === "alias" && !finished.has(ref)) {
          onPath.add(ref);
          stack.push([ref, aliasRefs(target)]);
          descended = true;
          break;
        }
      }
      if (!descended) {
        stack.pop();
        onPath.delete(name);
        finished.add(name);
      }
    }
  }
}

// A PascalCase name is an identifier whose first character is uppercase.
const PASCAL_CASE = new RegExp(`^(?=\\p{Uppercase})${IDENTIFIER_PATTERN}$`, "u");

/** Whether `value` is a plain object: its prototype is `Object.prototype` or `null`. */
function isPlainMapping(value: unknown): value is Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null) return false;
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/**
 * The type name diagnostics show for the host value `value`: `dict` for a plain
 * object, `list` for an array, `str`, `int` for a bigint, `float` for a number, `bool`,
 * `NoneType` for `null`, else its constructor's name (or `typeof`).
 */
export function typeName(value: unknown): string {
  if (isPlainMapping(value)) return "dict";
  if (Array.isArray(value)) return "list";
  switch (typeof value) {
    case "string":
      return "str";
    case "bigint":
      return "int";
    case "number":
      // The loader reads YAML ints as bigints, so a number here is always a YAML float.
      return "float";
    case "boolean":
      return "bool";
  }
  if (value === null) return "NoneType";
  if (value === undefined) return typeof value;
  const name: unknown = (value as { constructor?: { name?: unknown } }).constructor?.name;
  return typeof name === "string" ? name : typeof value;
}

/**
 * Parse each text field type of record typedef `name` that opens more `(` than it closes.
 *
 * Only such a field can be the front of a type a comma cut; any other field's mistake is
 * left for its own check.
 *
 * @throws TypeCheckError naming the first such field, whose type text doesn't parse.
 */
function checkCutFields(name: string, fields: Readonly<Record<string, unknown>>): void {
  for (const [field, written] of Object.entries(fields)) {
    if (typeof written !== "string") continue;
    if (written.split("(").length <= written.split(")").length) continue;
    try {
      parseType(written);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      throw typedefError(name, field, error, { cause: error });
    }
  }
}

/**
 * Build a `TypeRegistry` from a raw `typedefs:` mapping.
 *
 * A plain-object value is a RECORD (`RecordDef`; field type texts kept raw, in property
 * order). A `Literal(...)` string is an ENUM (`VariantDef`, tag-only). Any other string is
 * an ALIAS (`AliasDef`). An array is the dropped tagged-union spelling and is rejected: an
 * all-bare-tags array must be a `Literal(...)` enum; a payload array must be a `Union(...)`
 * of records. Names must be PascalCase and must not shadow a scalar or typing constructor.
 * Every typedef is then resolved, even one nothing uses, so a bad field type, an unknown
 * name, a recursive record, an alias cycle, or a bad `Union(...)` fails here.
 *
 * `raw` maps each name to its definition; `null` or `undefined` means none. `memoryTypes` names
 * the host memory types the typedefs resolve against, which the registry returned leaves out.
 *
 * Example: `readTypedefs({Buy: {action: "Literal(buy)"}, Sell: {action: "Literal(sell)"},
 * Decision: "Union(Buy, Sell)"})` registers two records and an alias to their union.
 *
 * @throws TypeCheckError when a name is not PascalCase or shadows a builtin (a built-in memory
 *   type included), a field name is
 *   empty or holds a Unicode surrogate, a field type is not a string, a value is a union
 *   array or of another type, a typedef nests more record and list levels than the type
 *   nesting limit or ends a longer chain of typedefs each naming the next, or a typedef
 *   does not resolve (an unknown constructor or name, bad bounds, a recursive record, an
 *   alias cycle, or a bad `Union(...)`). The message names the typedef and field, and
 *   `path` is `[typedef]` or `[typedef, field]` when the error belongs to one entry.
 */
export function readTypedefs(
  raw: Readonly<Record<string, unknown>> | null | undefined,
  memoryTypes: Iterable<string> = [],
): TypeRegistry {
  const registry = new Map<string, TypeDef>();
  for (const [name, value] of Object.entries(raw ?? {})) {
    if (!PASCAL_CASE.test(name)) {
      throw new TypeCheckError(
        `typedef name ${reprText(name)} must be PascalCase (an uppercase identifier)`,
        { path: [name] },
      );
    }
    if (BUILTIN_RECORD_NAMES.has(name)) {
      throw new TypeCheckError(
        `typedef name ${reprText(name)} shadows the built-in type ${reprText(name)}`,
        { path: [name] },
      );
    }
    if (BUILTIN_MEMORY_TYPES.has(name)) {
      throw new TypeCheckError(
        `typedef name ${reprText(name)} shadows the built-in memory type ${reprText(name)}`,
        { path: [name] },
      );
    }
    if (isShadow(name)) {
      throw new TypeCheckError(
        `typedef name ${reprText(name)} shadows a builtin scalar or typing constructor`,
        { path: [name] },
      );
    }
    if (isPlainMapping(value)) {
      const fields = new Map<string, string>();
      for (const [fname, ftype] of Object.entries(value)) {
        if (!fname) {
          throw new TypeCheckError(
            `typedef ${reprText(name)} field ${reprText(fname)}: field names must be ` +
              "non-empty strings",
            { path: [name, fname] },
          );
        }
        if (!encodesAsUtf8(fname)) {
          throw new TypeCheckError(
            `typedef ${reprText(name)} field ${reprText(fname)}: field names must not ` +
              "contain a Unicode surrogate",
            { path: [name, fname] },
          );
        }
        if (typeof ftype !== "string") {
          // A `{...}` mapping's comma cuts `Literal(a, b)` into a field `Literal(a` and a
          // null-valued field `b)`; report the cut type, whose parse error says why.
          checkCutFields(name, value);
          throw new TypeCheckError(
            `typedef ${reprText(name)} field ${reprText(fname)}: type must be a string ` +
              `expression (got ${typeName(ftype)})`,
            { path: [name, fname] },
          );
        }
        fields.set(fname, ftype);
      }
      registry.set(name, { kind: "record", fields });
    } else if (typeof value === "string") {
      let parsed: TypeExpr;
      try {
        parsed = parseType(value);
      } catch (error) {
        if (!(error instanceof TypeCheckError)) throw error;
        throw typedefError(name, null, error, { cause: error });
      }
      registry.set(
        name,
        parsed.kind === "enum"
          ? { kind: "variant", tags: parsed.tags }
          : { kind: "alias", target: parsed },
      );
    } else if (Array.isArray(value)) {
      if (value.every((member): member is string => typeof member === "string")) {
        // `Literal()` doesn't parse, so an empty sequence gets no example.
        const example = value.length > 0 ? ` (e.g. ${name}: ${literalLabel(value)})` : "";
        throw new TypeCheckError(
          `typedef ${reprText(name)}: a tag-only union must be a Literal(...) enum, not a ` +
            `sequence${example}`,
          { path: [name] },
        );
      }
      throw new TypeCheckError(
        `typedef ${reprText(name)}: a tagged/payload union sequence is not supported; ` +
          `declare each member as a record and write ${name}: Union(A, B, ...)`,
        { path: [name] },
      );
    } else {
      throw new TypeCheckError(
        `typedef ${reprText(name)}: unsupported definition of type ${typeName(value)}`,
        { path: [name] },
      );
    }
  }
  checkAliasCycles(registry);
  // Resolved beside the host's memory types, so a typedef naming one is refused as a memory.
  checkTypedefs(withMemoryTypes(registry, memoryTypes));
  return registry;
}

/**
 * Return `registry` with the host memory types `names` added, so a type names each as written
 * (`TradeJournal`, `Optional(TradeJournal)`). A built-in memory type's name is skipped, as a class
 * that implements one needs no entry, and so is a name `registry` already holds as a typedef,
 * which keeps its meaning: the loader refuses a flow input that names it.
 *
 * @throws TypeCheckError, at `[name]`, when a name is not PascalCase or shadows a built-in name.
 */
export function withMemoryTypes(registry: TypeRegistry, names: Iterable<string>): TypeRegistry {
  const added = new Map(registry);
  for (const name of names) {
    if (BUILTIN_MEMORY_TYPES.has(name)) continue;
    const fail = (message: string): never => {
      throw new TypeCheckError(message, { path: [name] });
    };
    if (!PASCAL_CASE.test(name)) {
      fail(`memory type name ${reprText(name)} must be PascalCase (an uppercase identifier)`);
    }
    if (BUILTIN_RECORD_NAMES.has(name)) {
      fail(`memory type name ${reprText(name)} shadows the built-in type ${reprText(name)}`);
    }
    if (isShadow(name)) fail(`memory type name ${reprText(name)} shadows a builtin scalar or typing constructor`);
    if (registry.get(name)?.kind === "memory" || !registry.has(name)) added.set(name, { kind: "memory" });
  }
  return added;
}
