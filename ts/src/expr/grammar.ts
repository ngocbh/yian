/**
 * One unified `${...}` expression grammar: parse only, no evaluation here.
 *
 * This one grammar parses binding coalesce/default expressions, `when:` boolean and
 * arithmetic expressions, and prompt builtin calls into one tree shaped like Lark's, so code
 * that walks Python yian's trees ports line for line. Evaluation lives in `expressions.ts`.
 *
 * ## What it parses
 *
 * - References: bare `a`, dotted `a.b.c`, and, for the runtime graph-expansion separators,
 *   segments containing `#` / `/` (`node#0.output`, `def/child.output`).
 * - Arithmetic (`+ - * / %` and `**`), unary minus, comparisons (`== != < <= > >=`),
 *   `and` / `or` / `not`, `in` / `not in`. A chain of same-precedence operators is one flat
 *   node holding its operator tokens between the operands: `a + b - c` is one `sum`.
 * - List literals (`[1, "x"]`) and parenthesised sub-expressions.
 * - Pure builtin calls (`upper(name)`, `join(items, ", ")`), with dotted access on the result
 *   (`fn(x).field`), keyword arguments (`f(a=1)`), constructor calls `T(x)` / `T(a=...)`, and
 *   the `isinstance(x, T)` special form, all parsed alike.
 * - Coalesce (`a | b | c`), default (`a :- d`), required (`a :? "msg"`).
 * - Literals: numbers (ASCII digits, an optional fraction, no exponent), quoted strings (no
 *   escapes), `true` / `false`, `null` / `none`, and `inf`.
 *
 * A `refcall` is a `NAME`, dotted `trailer`s, then an optional `call_suffix` and more trailers.
 * No suffix is a reference; a suffix makes a builtin call whose callee is the leading `NAME`,
 * and trailers after the parentheses are dotted access on the result.
 *
 * ## Why `-` is not in `NAME` (and `#` / `/` are)
 *
 * `NAME` is `[A-Za-z_][A-Za-z0-9_#/]*`. With `-` in it, `a - b` would lex as the one name `a-b`
 * and subtraction would break, so a hyphenated identifier is unavailable inside an expression.
 * `#` and `/` are in it because runtime graph expansion mints segments like `node#0` and
 * `def/child`.
 *
 * ## Keywords are contextual
 *
 * The lexer offers only the tokens that can come next, as Lark's contextual lexer does. Where
 * an operand is expected, `and`, `or` and `in` lex as names (`and(x)` is a call, `fn(and=x)` a
 * keyword argument), and `not` does too after a unary minus or a comparison or arithmetic
 * operator. After a `.` every keyword is a plain field name (`record.output.in`, `x.inf`).
 * The post-parse checks reject the operator words as reference roots.
 *
 * ## Why the parser is iterative
 *
 * Expressions nest deeply (thousands of parentheses must parse, and thousands of nested
 * prefixes or calls must fail with the nesting message), which a recursive parser would turn
 * into a stack overflow. The parser keeps its pending groups, lists, calls and operators on an
 * explicit stack instead.
 */

import { reprText } from "../diagnostics.ts";
import { ExpressionError } from "./expressions.ts";

/** A terminal of the parse: its type, its source text, and where it sits in the text. */
export class Token {
  readonly type: string;
  /** The source text; a `NUMBER` stays text and a `STRING` keeps its quotes. */
  readonly value: string;
  /** The UTF-16 offset of the first character in the text, or 0 for a token built elsewhere. */
  readonly startPos: number;
  /** The UTF-16 offset after the last character, or 0 for a token built elsewhere. */
  readonly endPos: number;

  constructor(type: string, value: string, startPos = 0, endPos = 0) {
    this.type = type;
    this.value = value;
    this.startPos = startPos;
    this.endPos = endPos;
  }

  /** Return the source text. */
  toString(): string {
    return this.value;
  }
}

/** A rule node of the parse: the rule's name and its children. */
export class Tree {
  readonly data: string;
  readonly children: ExprNode[];

  constructor(data: string, children: ExprNode[]) {
    this.data = data;
    this.children = children;
  }
}

/** A parsed expression or a part of one. */
export type ExprNode = Tree | Token;

/**
 * Return every subtree of `tree`, itself included, each once, in Lark's `iter_subtrees` order:
 * a deeper node before a shallower one, and the left one first at the same depth.
 */
export function iterSubtrees(tree: Tree): Tree[] {
  // Lark walks breadth first taking children right to left, then reverses the walk.
  const queue: Tree[] = [tree];
  const seen = new Set<Tree>(queue);
  for (let i = 0; i < queue.length; i++) {
    const children = queue[i]!.children;
    for (let j = children.length - 1; j >= 0; j--) {
      const child = children[j];
      if (child instanceof Tree && !seen.has(child)) {
        seen.add(child);
        queue.push(child);
      }
    }
  }
  return queue.reverse();
}

/** Precedence levels, low to high: a pending operator above the incoming one completes first. */
const LEVEL = {
  COALESCE: 1,
  DEFAULT: 2,
  OR: 3,
  AND: 4,
  NOT: 5,
  COMPARE: 6,
  SUM: 7,
  PRODUCT: 8,
  UNARY: 9,
  POWER: 10,
} as const;

/** A token type and its sticky pattern. Punctuation is typed by its text. */
type Terminal = readonly [type: string, pattern: RegExp];

const terminal = (type: string, source: string): Terminal => [type, new RegExp(source, "y")];
const punctuation = (text: string): Terminal => [text, new RegExp(text.replace(/[.()[\]|?]/g, "\\$&"), "y")];
// Keywords need only a trailing word boundary, ASCII as every other terminal.
const keyword = (type: string, words: string): Terminal => terminal(type, `(?:${words})(?![A-Za-z0-9_])`);

const NOT_IN = keyword("NOT_IN", "not\\s+in");
const NOT = terminal("NOT", "not(?![A-Za-z0-9_#/])");
const IN = keyword("IN", "in");
const AND = keyword("AND", "and");
const OR = keyword("OR", "or");
const BOOL = keyword("BOOL", "true|false");
const NULL = keyword("NULL", "null|none");
const INF = keyword("INF", "inf");
const NAME = terminal("NAME", "[A-Za-z_][A-Za-z0-9_#/]*");
const NUMBER = terminal("NUMBER", "[0-9]+(?:\\.[0-9]+)?");
const STRING = terminal("STRING", `"[^"]*"|'[^']*'`);
const POW_OP = terminal("POW_OP", "\\*\\*");
const MUL_OP = terminal("MUL_OP", "[*/%]");
const ADD_OP = terminal("ADD_OP", "[+-]");
const COMP_OP = terminal("COMP_OP", "==|!=|<=|>=|<|>");
const DOT = punctuation(".");
const COMMA = punctuation(",");
const LPAR = punctuation("(");
const RPAR = punctuation(")");
const LSQB = punctuation("[");
const RSQB = punctuation("]");
const EQUAL = punctuation("=");
const VBAR = punctuation("|");
const MINUS = punctuation("-");
const DEFAULT = punctuation(":-");
const REQUIRED = punctuation(":?");

// Where an operand is expected. The first match wins, so the keywords beat NAME.
const OPERAND: readonly Terminal[] = [LPAR, LSQB, MINUS, NOT, BOOL, NULL, INF, NAME, NUMBER, STRING];
const OPERAND_WITHOUT_NOT = OPERAND.filter((t) => t !== NOT);
// After an operand. NOT_IN before IN and `**` before `*`.
const OPERATOR: readonly Terminal[] = [
  NOT_IN, IN, AND, OR, VBAR, DEFAULT, REQUIRED, POW_OP, MUL_OP, ADD_OP, COMP_OP, RPAR, RSQB, COMMA,
];
// Every terminal, to name what a parse error met.
const ALL: readonly Terminal[] = [
  NOT_IN, NOT, IN, AND, OR, BOOL, NULL, INF, NAME, NUMBER, STRING, POW_OP, MUL_OP, ADD_OP, COMP_OP,
  DOT, COMMA, LPAR, RPAR, LSQB, RSQB, EQUAL, VBAR, DEFAULT, REQUIRED,
];

const WHITESPACE = /[ \t\f\r\n]+/y;

/** The text could not be parsed; the message says what was met where. */
class ParseError extends Error {
  override name = "ParseError";
}

/** Return the first of `terminals` matching `text` at `pos` as a token, or null. */
function lexAt(text: string, pos: number, terminals: readonly Terminal[]): Token | null {
  for (const [type, pattern] of terminals) {
    pattern.lastIndex = pos;
    const match = pattern.exec(text);
    if (match) return new Token(type, match[0], pos, pos + match[0].length);
  }
  return null;
}

/** A position in the text being parsed, and the reads the parser makes there. */
class Cursor {
  readonly text: string;
  pos = 0;

  constructor(text: string) {
    this.text = text;
  }

  /** Move past any whitespace. */
  skip(): void {
    WHITESPACE.lastIndex = this.pos;
    if (WHITESPACE.test(this.text)) this.pos = WHITESPACE.lastIndex;
  }

  /** Skip whitespace and return whether the text ends here. */
  atEnd(): boolean {
    this.skip();
    return this.pos >= this.text.length;
  }

  /** Skip whitespace and consume the first of `terminals` that matches, or fail here. */
  next(terminals: readonly Terminal[]): Token {
    this.skip();
    const token = lexAt(this.text, this.pos, terminals) ?? this.fail(this.pos);
    this.pos = token.endPos;
    return token;
  }

  /** Skip whitespace and consume `literal` if it comes next. */
  accept(literal: string): boolean {
    this.skip();
    if (!this.text.startsWith(literal, this.pos)) return false;
    this.pos += literal.length;
    return true;
  }

  /** Skip whitespace and consume a keyword argument's `=` (not `==`) if it comes next. */
  acceptLabel(): boolean {
    this.skip();
    if (this.text[this.pos] !== "=" || this.text[this.pos + 1] === "=") return false;
    this.pos += 1;
    return true;
  }

  /** Throw a `ParseError` naming what is at `pos` and its line and column. */
  fail(pos: number): never {
    const { text } = this;
    let what = "end of expression";
    if (pos < text.length) {
      const token = lexAt(text, pos, ALL);
      const ch = String.fromCodePoint(text.codePointAt(pos)!);
      what = token ? reprText(token.value) : `character ${reprText(ch)}`;
    }
    // Lines break at LF, CRLF and CR; columns count UTF-16 code units from 1.
    let line = 1;
    let lineStart = 0;
    for (let i = 0; i < pos; i++) {
      if (text[i] === "\n" || (text[i] === "\r" && text[i + 1] !== "\n")) {
        line += 1;
        lineStart = i + 1;
      }
    }
    throw new ParseError(`unexpected ${what} at line ${line}, column ${pos - lineStart + 1}`);
  }
}

/** A construct the parser has opened and not finished. */
type Frame =
  | { kind: "group" }
  | { kind: "list"; items: ExprNode[] }
  | { kind: "call"; children: ExprNode[]; args: Tree[]; label: Token | null }
  | { kind: "prefix"; level: typeof LEVEL.NOT | typeof LEVEL.UNARY; token: Token };

type CallFrame = Extract<Frame, { kind: "call" }>;

/** Read `.NAME` trailers into a refcall's `children`; any keyword is a plain name here. */
function readTrailers(cursor: Cursor, children: ExprNode[]): void {
  while (cursor.accept(".")) children.push(new Tree("trailer", [cursor.next([NAME])]));
}

/** Close `call`'s argument list, read the trailers after it, and return the refcall. */
function finishCall(cursor: Cursor, call: CallFrame): Tree {
  call.children.push(new Tree("call_suffix", call.args));
  readTrailers(cursor, call.children);
  return new Tree("refcall", call.children);
}

/** Add `value`, under the pending label if any, as `call`'s next argument. */
function addArg(call: CallFrame, value: ExprNode): void {
  call.args.push(new Tree("arg", call.label ? [call.label, value] : [value]));
  call.label = null;
}

/** Complete every prefix frame on top of `stack` above `level` around `value`; return the result. */
function reduce(stack: Frame[], value: ExprNode, level: number): ExprNode {
  for (let top = stack.at(-1); top?.kind === "prefix" && top.level > level; top = stack.at(-1)) {
    stack.pop();
    value = top.level === LEVEL.NOT ? new Tree("negate", [top.token, value]) : new Tree("neg", [value]);
  }
  return value;
}

/** Parse `text` into a tree, or throw a `ParseError`; no post-parse checks. */
function parseRaw(text: string): ExprNode {
  const cursor = new Cursor(text);
  const stack: Frame[] = [];
  // The operand just read; null while one is expected.
  let value: ExprNode | null = null;
  for (;;) {
    const top = stack.at(-1);
    if (value === null) {
      const offersNot = !(top?.kind === "prefix" && top.level === LEVEL.UNARY);
      const token = cursor.next(offersNot ? OPERAND : OPERAND_WITHOUT_NOT);
      switch (token.type) {
        case "(":
          stack.push({ kind: "group" });
          break;
        case "[":
          if (cursor.accept("]")) value = new Tree("list_lit", []);
          else stack.push({ kind: "list", items: [] });
          break;
        case "-":
          stack.push({ kind: "prefix", level: LEVEL.UNARY, token });
          break;
        case "NOT":
          stack.push({ kind: "prefix", level: LEVEL.NOT, token });
          break;
        case "NAME": {
          if (top?.kind === "call" && top.label === null && cursor.acceptLabel()) {
            top.label = token;
            break;
          }
          const children: ExprNode[] = [token];
          readTrailers(cursor, children);
          if (!cursor.accept("(")) {
            value = new Tree("refcall", children);
            break;
          }
          const call: CallFrame = { kind: "call", children, args: [], label: null };
          if (cursor.accept(")")) value = finishCall(cursor, call);
          else stack.push(call);
          break;
        }
        default:
          value = token;
      }
      continue;
    }
    if (cursor.atEnd()) {
      value = reduce(stack, value, 0);
      if (stack.length === 0) return value;
      cursor.fail(text.length);
    }
    const token = cursor.next(OPERATOR);
    if (token.type !== ")" && token.type !== "]" && token.type !== ",") {
      throw new Error(`the operator ${reprText(token.value)} is not supported yet`);
    }
    value = reduce(stack, value, 0);
    const frame = stack.at(-1);
    if (token.type === ")" && frame?.kind === "group") {
      stack.pop();
    } else if (token.type === ")" && frame?.kind === "call") {
      stack.pop();
      addArg(frame, value);
      value = finishCall(cursor, frame);
    } else if (token.type === "]" && frame?.kind === "list") {
      stack.pop();
      frame.items.push(value);
      value = new Tree("list_lit", frame.items);
    } else if (token.type === "," && frame?.kind === "list") {
      frame.items.push(value);
      value = null;
    } else if (token.type === "," && frame?.kind === "call") {
      addArg(frame, value);
      value = null;
    } else {
      cursor.fail(token.startPos);
    }
  }
}

/**
 * Parse one unified `${...}` expression into a tree (parse only).
 *
 * This does no evaluation; it is the shared front end for the evaluator in `expressions.ts`.
 * A lone literal (a `NUMBER`, `STRING`, `BOOL`, `NULL` or `INF`) comes back as a bare `Token`,
 * since a level with one operand makes no node.
 *
 * @throws {ExpressionError} If `text` does not parse:
 *   `could not parse expression '<text>': unexpected <what> at line L, column C`.
 */
export function parseExpr(text: string): ExprNode {
  try {
    return parseRaw(text);
  } catch (error) {
    if (!(error instanceof ParseError)) throw error;
    throw new ExpressionError(`could not parse expression ${reprText(text)}: ${error.message}`, {
      cause: error,
    });
  }
}
