/**
 * Authored type text → `TypeExpr` AST → resolved `Type`.
 *
 * Parses an authored type expression into a `TypeExpr` AST, the bridge from the authoring
 * contract to the typed value system (`values.ts`), and prints a parsed type back in the
 * parenthesized spelling. Compose and compile type resolution consume it; runtime receives
 * only resolved `Type` values.
 */

import { reprText } from "../diagnostics.ts";
import {
  type Bound,
  type Bounds,
  BUILTIN_CONSTANTS,
  boundText,
  literalLabel,
  NAME_PATTERN,
  TypeCheckError,
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

/** `dict(K, V)`: resolves to a lenient object; key and value types are not kept. */
export interface DictExpr {
  readonly kind: "dict";
}

/** The authored `Union(...)` members, in authored order. */
export interface UnionExpr {
  readonly kind: "union";
  readonly members: readonly TypeExpr[];
}

/** A parsed type expression, still unresolved against any registry. */
export type TypeExpr =
  | ScalarExpr
  | ListExpr
  | RefExpr
  | OptionalExpr
  | EnumExpr
  | DictExpr
  | UnionExpr;

const SCALAR_KEYWORDS: ReadonlySet<string> = new Set([
  "str", "int", "float", "bool", "date", "datetime", "object", "file", "None", "none", "Any",
]);

// Names a typedef must not shadow (typing constructors and the generic builtins).
const TYPING_CONSTRUCTORS: ReadonlySet<string> = new Set([
  "Optional", "Literal", "Union", "List", "Dict", "list", "dict", "Any",
]);

/**
 * Names a type expression may start with besides typedefs, derived from the keyword sets
 * so they cannot drift. `topics` is a retired alias, so it is not listed.
 */
export const BUILTIN_TYPE_NAMES: ReadonlySet<string> = new Set([
  ...SCALAR_KEYWORDS,
  ...TYPING_CONSTRUCTORS,
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

/**
 * The deepest constructor nesting a type may have.
 *
 * Becomes `current().maxTypeNesting` once the defaults module exists.
 */
function typeNestingLimit(): number {
  return 100;
}

/** Return whether `text` encodes as UTF-8, i.e. holds no lone Unicode surrogate. */
export function encodesAsUtf8(text: string): boolean {
  return text.isWellFormed();
}

/**
 * Parse an authored type expression into a `TypeExpr` AST.
 *
 * The vocabulary is `str`/`int`/`float`/`bool`/`Any`/`date`/`datetime`/`object`/`file`/
 * `None`, where `Any` is any JSON value except null and `object` is a JSON object,
 * inclusive ranges `int(1..12)`/`float(0.0..1.0)` (either end may be open), and the
 * constructors `list(X)`/`Optional(X)`/`dict(K, V)`/`Literal(a, "b c")`/
 * `Union(A, B, ...)`, plus the `topics` alias. A bare name that is neither a scalar nor a
 * constructor is a registry reference. `Union(...)` members are only checked to be
 * records when the type is resolved. A quoted `Literal` member takes JSON escapes.
 *
 * Example: `parseType("Union(Buy, Sell)")` gives
 * `{kind: "union", members: [{kind: "ref", name: "Buy"}, {kind: "ref", name: "Sell"}]}`.
 *
 * @throws TypeCheckError when the text is not a valid expression (including unbalanced
 *   parentheses, a lone Unicode surrogate, or an empty `Literal` member), uses an unknown
 *   constructor or the lowercase `any`, gives a numeric range with no end, no `..`, or as
 *   two arguments, spells a union as `A | B` (the hint suggests `Union(A, B)`), or lists
 *   `None` as a `Union(...)` member (the hint suggests `Optional(Union(...))`).
 */
export function parseType(text: string): TypeExpr {
  const src = text.trim();
  if (!encodesAsUtf8(src)) {
    throw new TypeCheckError(
      `type expression ${reprText(src)} contains a Unicode surrogate ` +
        "(types must be valid UTF-8 text)",
    );
  }
  return new TypeParser(src).parse();
}

// NUMBER needs a digit after a dot, so `0..1` reads as `0`, `..`, `1`. Its digits are
// ASCII only, so `int(١..٣)` is not a range. An escape is a backslash and any character
// but LF, as Python's `\\.`; JavaScript's `.` would also refuse CR, U+2028 and U+2029.
const TYPE_TOKEN = new RegExp(
  String.raw`\s*(?:(?<number>[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)` +
    `|(?<name>${NAME_PATTERN})` +
    String.raw`|(?<string>"(?:[^"\\]|\\[^\n])*"|'(?:[^'\\]|\\[^\n])*')` +
    String.raw`|(?<punct>\.\.|[(),|]))`,
  "uy",
);

const TOKEN_KINDS = ["number", "name", "string", "punct"] as const;

type TokenKind = (typeof TOKEN_KINDS)[number];

interface Token {
  readonly kind: TokenKind;
  readonly text: string;
}

/** Recursive-descent parser for the parenthesized type grammar. */
class TypeParser {
  private readonly src: string;
  private readonly tokens: Token[] = [];
  private index = 0;
  private depth = 0;
  private readonly maxDepth = typeNestingLimit();

  constructor(src: string) {
    this.src = src;
    let pos = 0;
    while (pos < src.length) {
      TYPE_TOKEN.lastIndex = pos;
      const match = TYPE_TOKEN.exec(src);
      if (match === null) {
        // An unclosed quote often means a `{...}` mapping's comma split the type.
        if (this.opensUnclosed()) this.unbalanced();
        this.malformed();
      }
      const groups = match.groups!;
      const kind = TOKEN_KINDS.find((name) => groups[name] !== undefined)!;
      this.tokens.push({ kind, text: groups[kind]! });
      pos = TYPE_TOKEN.lastIndex;
    }
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
    const suffix = detail ? `: ${detail}` : "";
    throw new TypeCheckError(`malformed type expression ${reprText(this.src)}${suffix}`);
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
    if (this.depth > this.maxDepth) {
      const head = Array.from(this.src).slice(0, 40).join("");
      throw new TypeCheckError(
        `type nested too deeply in ${reprText(head)}... (at most ${this.maxDepth} levels)`,
      );
    }
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
      case "Optional":
        return { kind: "optional", inner: this.oneType(name) };
      case "Literal":
        return { kind: "enum", tags: this.literalTags() };
      case "dict":
      case "Dict":
        this.args(name, 2);
        return { kind: "dict" };
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
    const bounds: Bounds = [low, high];
    try {
      validateNumericBounds(SCALAR_KIND.get(ctor)!, bounds);
    } catch (error) {
      if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
      throw new TypeCheckError(`invalid bounds in ${reprText(this.src)}: ${error.message}`, {
        cause: error,
      });
    }
    return { kind: "scalar", name: ctor, bounds };
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

/** Read a bare type name: a scalar keyword, an alias, or a registry reference. */
function namedType(name: string): TypeExpr {
  if (name === "any") {
    // `any` stays a builtin name (the expression function), so it never reads as a
    // registry reference.
    throw new TypeCheckError("unknown type 'any'; write Any");
  }
  if (name === "topics") return { kind: "list", element: { kind: "scalar", name: "str" } };
  // The historical lowercase token.
  if (name === "none") return { kind: "scalar", name: "None" };
  if (SCALAR_KEYWORDS.has(name)) return { kind: "scalar", name: name as ScalarName };
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
    throw new TypeCheckError(`malformed type expression ${reprText(src)}: ${detail}`);
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

/**
 * Render a parsed type in the parenthesized spelling `parseType` reads back.
 *
 * A float range end prints as Python's `repr` of the float (`1.0`, `1e-05`) and an int
 * end as its digits, as labels spell them. A `dict(K, V)` renders as `dict`, since its
 * key and value types are not kept.
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
    case "optional":
      return `Optional(${formatTypeExpr(t.inner)})`;
    case "enum":
      return literalLabel(t.tags);
    case "union":
      return "Union(" + t.members.map(formatTypeExpr).join(", ") + ")";
    case "dict":
      return "dict";
  }
}
