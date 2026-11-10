/**
 * One unified `${...}` expression grammar: parse only, no evaluation here.
 *
 * This one grammar parses binding coalesce/default expressions, `when:` boolean and
 * arithmetic expressions, and prompt builtin calls into one tree shaped like Lark's.
 * Evaluation lives in `expressions.ts`.
 *
 * ## What it parses
 *
 * - References: bare `a`, dotted `a.b.c`, optional steps `a?.b` (null when `a` is null or
 *   missing), and, for the runtime graph-expansion separators, segments containing `#` / `/`
 *   (`node#0.output`, `def/child.output`).
 * - Arithmetic (`+ - * / %` and `**`), unary minus, comparisons (`== != < <= > >=`),
 *   `and` / `or` / `not`, `in` / `not in`. A chain of same-precedence operators is one flat
 *   node holding its operator tokens between the operands: `a + b - c` is one `sum`.
 * - List literals (`[1, "x"]`) and parenthesised sub-expressions.
 * - Subscripts after any operand: an index `xs[i]`, or a slice `xs[a:b]` with either bound
 *   omitted (`xs[1:]`, `xs[:-1]`, `xs[:]`) and no step.
 * - List comprehensions: `[elem for x in xs]`, then any mix of `for x in ys` and
 *   `if cond` clauses. A target is one bare name; each iterable and condition stops before the
 *   escapes, so an escape there needs parentheses.
 * - Pure builtin calls (`upper(name)`, `join(items, ", ")`), with dotted access on the result
 *   (`fn(x).field`), keyword arguments (`f(a=1)`), constructor calls `T(x)` / `T(a=...)`, and
 *   the `isinstance(x, T)` special form, all parsed alike.
 * - Coalesce (`a | b | c`), default (`a :- d`), required (`a :? "msg"`).
 * - Conditionals: `a if c else b`, the lowest level, as in Python; a chain groups to the right.
 * - Literals: numbers (ASCII digits, an optional fraction, no exponent), quoted strings (no
 *   escapes), `true` / `false`, `null` / `none`, and `inf`.
 *
 * A `refcall` is a `NAME`, dotted `trailer`s, then an optional `call_suffix` and more trailers.
 * No suffix is a reference; a suffix makes a builtin call whose callee is the leading `NAME`,
 * and trailers after the parentheses are dotted access on the result.
 *
 * ## Tree shape
 *
 * Precedence, low to high, with each level's node:
 *
 * - `a if c else b`: `if_expr[a, c, b]`, in source order, no tokens. `a` and `b` take every
 *   level above it (`x :- y if c else z` is `if_expr[default_expr, c, z]`), and so does `c`,
 *   up to its `else`, but `c` can't be a conditional unless parenthesized. It counts one nesting
 *   level.
 * - `a | b | c`: `coalesce[a, b, c]`, no tokens; present only when a `|` appears.
 * - `a :- d`, `a :? m`: `default_expr[a, d]`, `required_expr[a, m]`, no token; they don't chain.
 * - `or`, `and`: `or_expr` / `and_expr`, with the `OR` / `AND` tokens between the operands.
 * - `not x`: `negate[NOT, x]`.
 * - Comparisons: `compare[a, COMP_OP, b]`, `compare_in[a, IN, b]`, `compare_notin[a, NOT_IN, b]`;
 *   they don't chain.
 * - `+ -`, `* / %`: `sum`, `product`, flat, with `ADD_OP` / `MUL_OP` tokens between the operands.
 * - Unary `-x`: `neg[x]`, no token.
 * - `**`: `power`, flat, with `POW_OP` tokens between the operands.
 * - A postfix subscript: `x[i]` is `subscript[x, index[i]]`, and `x[a:b]` is
 *   `subscript[x, slice[a, b]]`, where an omitted bound is `omitted[]`. It binds tighter than
 *   unary minus and `**`: `-xs[1:]` is `neg[subscript]`. It counts one nesting level, like a
 *   call; `index`, `slice` and `omitted` add none.
 * - A list comprehension: `list_comp[elem, clause, ...]`, clauses in source order, each
 *   `comp_for[NAME, iterable, SOURCE]` (`SOURCE` holds the iterable's source text, for runtime
 *   messages) or `comp_if[cond]`. It counts one nesting level, like a list; its clauses add none.
 *   After the scoping pass a variable read is `comp_var[NAME, trailer*]`, which adds none, like a
 *   plain reference.
 *
 * A chain of one level is one node, and author parentheses start a new one. A level with one
 * operand makes no node and parentheses make none, so a lone literal is a bare `Token`. The
 * operands: `list_lit[items]`, `refcall[NAME, trailer*, call_suffix?, trailer*]`,
 * `trailer[NAME, QDOT?]` (a `?.` step adds the `QDOT`), `call_suffix[arg*]`, and `arg[NAME?, expr]`
 * (a keyword argument's label first).
 *
 * Token types: `NAME`, `NUMBER` (text; no `.` is an int, with `.` a float), `STRING` (keeps its
 * quotes), `BOOL`, `NULL` (`null` / `none`), `INF`, `NOT`, `NOT_IN` (keeps its source gap), `IN`,
 * `AND`, `OR`, `COMP_OP`, `ADD_OP`, `MUL_OP`, `POW_OP`, `QDOT` (`?.`), and `SOURCE` (a
 * comprehension iterable's text). `startPos` / `endPos` are UTF-16 offsets.
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
 * `for`, `if` and `else` are keywords only where an operator is expected, after a complete
 * operand at any bracket depth: in a list's brackets `for` after the first element starts a
 * comprehension; `if` after a comprehension's iterable or condition adds a filter, and anywhere
 * else starts a conditional, which `else` continues. A misplaced one is a parse error that names
 * the fix. Where an operand is expected they are names, as before.
 *
 * The colon rule: where the nearest enclosing bracket is a subscript's `[` (pending operators,
 * prefixes and a conditional's else branch don't count; a conditional's test, which `else` ends,
 * does), the lexer tries `:` before `:-` and `:?`, so `xs[a:-b]` and
 * `xs[n - 1:-1]` are slices. Inside parentheses, a call or a list within a bound, `:-` and `:?`
 * are the escapes as elsewhere, so a default in a bound is written `xs[(n :- 0):]`.
 *
 * ## Comprehension scoping
 *
 * Every parse ends with a pass that turns each read of a comprehension variable into a
 * `comp_var`, so the walkers that look for `refcall` reads (edges, name checks, guards, CASE
 * renames) skip variables with no change and still see the outer reads around them. Scoping is
 * lexical: a variable is in scope in the elem and in the clauses after its own
 * `for`, nested comprehensions included, and the first iterable runs in the enclosing scope. A
 * variable may share a name with a node, an input, a typedef or a builtin; a call is never a
 * variable read, nor is `isinstance`'s type name. The parser refuses a target that is a keyword,
 * `input`, `system`, `item`, `output` or starts with `__` (generated locals use that prefix), and
 * a `for` that rebinds a name already bound in its own or an enclosing comprehension.
 *
 * ## Why the parser is iterative
 *
 * Expressions nest deeply (thousands of parentheses must parse, and thousands of nested
 * prefixes or calls must fail with the nesting message), which a recursive parser would turn
 * into a stack overflow. The parser keeps its pending groups, lists, calls, subscripts,
 * comprehensions, conditionals and operators on an explicit stack instead.
 */

import { current } from "../defaults.ts";
import { reprText } from "../diagnostics.ts";

/** An expression could not be parsed or evaluated. */
export class ExpressionError extends RangeError {
  override name = "ExpressionError";
}

/**
 * The type-test special form `isinstance(x, T)`: its second argument is a type name
 * looked up in the node's type table, never a reference, so the evaluator, the reference
 * walkers and rewriters, and type inference treat only its first argument as an expression.
 */
export const ISINSTANCE = "isinstance";

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

/**
 * Return every subtree of `tree`, itself included, in Lark's `iter_subtrees_topdown` order: each
 * node before its children, and the children left to right, as a depth-first walk.
 */
export function iterSubtreesTopdown(tree: Tree): Tree[] {
  // Lark keeps a stack and pushes each node's children reversed, so the leftmost pops first.
  const out: Tree[] = [];
  const stack: ExprNode[] = [tree];
  while (stack.length > 0) {
    const node = stack.pop()!;
    if (!(node instanceof Tree)) continue;
    out.push(node);
    for (let i = node.children.length - 1; i >= 0; i--) stack.push(node.children[i]!);
  }
  return out;
}

/** Precedence levels, low to high: a pending operator above the incoming one completes first. */
const LEVEL = {
  COND: 1,
  COALESCE: 2,
  DEFAULT: 3,
  OR: 4,
  AND: 5,
  NOT: 6,
  COMPARE: 7,
  SUM: 8,
  PRODUCT: 9,
  UNARY: 10,
  POWER: 11,
} as const;

type Level = (typeof LEVEL)[keyof typeof LEVEL];

/** A token type and its sticky pattern. Punctuation is typed by its text. */
type Terminal = readonly [type: string, pattern: RegExp];

const terminal = (type: string, source: string): Terminal => [type, new RegExp(source, "y")];
const punctuation = (text: string): Terminal => [text, new RegExp(text.replace(/[.()[\]|?]/g, "\\$&"), "y")];
// Keywords need only a trailing word boundary, ASCII as every other terminal.
const keyword = (type: string, words: string): Terminal => terminal(type, `(?:${words})(?![A-Za-z0-9_])`);

const NOT_IN = keyword("NOT_IN", "not\\s+in");
// Not `keyword()`: NOT also refuses a `#` or `/` next, so `not#0` and `not/child` stay names.
const NOT = terminal("NOT", "not(?![A-Za-z0-9_#/])");
const IN = keyword("IN", "in");
const AND = keyword("AND", "and");
const OR = keyword("OR", "or");
const BOOL = keyword("BOOL", "true|false");
const NULL = keyword("NULL", "null|none");
const INF = keyword("INF", "inf");
const FOR = keyword("FOR", "for");
const IF = keyword("IF", "if");
const ELSE = keyword("ELSE", "else");
const NAME = terminal("NAME", "[A-Za-z_][A-Za-z0-9_#/]*");
const NUMBER = terminal("NUMBER", "[0-9]+(?:\\.[0-9]+)?");
const STRING = terminal("STRING", `"[^"]*"|'[^']*'`);
const POW_OP = terminal("POW_OP", "\\*\\*");
const MUL_OP = terminal("MUL_OP", "[*/%]");
const ADD_OP = terminal("ADD_OP", "[+-]");
const COMP_OP = terminal("COMP_OP", "==|!=|<=|>=|<|>");
const DOT = punctuation(".");
const QDOT = punctuation("?.");
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
const COLON = punctuation(":");

// Where an operand is expected. The first match wins, so the keywords beat NAME.
const OPERAND: readonly Terminal[] = [LPAR, LSQB, MINUS, NOT, BOOL, NULL, INF, NAME, NUMBER, STRING];
const OPERAND_WITHOUT_NOT = OPERAND.filter((t) => t !== NOT);
// After an operand. NOT_IN before IN and `**` before `*`.
const OPERATOR: readonly Terminal[] = [
  NOT_IN, IN, AND, OR, FOR, IF, ELSE, VBAR, DEFAULT, REQUIRED, POW_OP, MUL_OP, ADD_OP, COMP_OP, RPAR, RSQB, COMMA, LSQB,
];
// After an operand directly inside a subscript's brackets, where `:` ends a bound before `:-`/`:?`.
const OPERATOR_IN_SLICE: readonly Terminal[] = [COLON, ...OPERATOR];
// Every terminal, to name what a parse error met.
const ALL: readonly Terminal[] = [
  NOT_IN, NOT, IN, AND, OR, FOR, IF, ELSE, BOOL, NULL, INF, NAME, NUMBER, STRING, POW_OP, MUL_OP, ADD_OP, COMP_OP,
  DOT, QDOT, COMMA, LPAR, RPAR, LSQB, RSQB, EQUAL, VBAR, DEFAULT, REQUIRED, COLON,
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

  /** Skip whitespace and return whether `literal` comes next, without consuming it. */
  peek(literal: string): boolean {
    this.skip();
    return this.text.startsWith(literal, this.pos);
  }

  /** Skip whitespace and consume a keyword argument's `=` (not `==`) if it comes next. */
  acceptLabel(): boolean {
    this.skip();
    if (this.text[this.pos] !== "=" || this.text[this.pos + 1] === "=") return false;
    this.pos += 1;
    return true;
  }

  /**
   * Throw a `ParseError` with `message`, or else naming what is at `pos`, and the line and column
   * of `pos`.
   */
  fail(pos: number, message?: string): never {
    const { text } = this;
    // Lines break at LF, CRLF and CR; columns count UTF-16 code units from 1.
    let line = 1;
    let lineStart = 0;
    for (let i = 0; i < pos; i++) {
      if (text[i] === "\n" || (text[i] === "\r" && text[i + 1] !== "\n")) {
        line += 1;
        lineStart = i + 1;
      }
    }
    const where = `at line ${line}, column ${pos - lineStart + 1}`;
    if (message !== undefined) throw new ParseError(`${message} ${where}`);
    let what = "end of expression";
    if (pos < text.length) {
      const token = lexAt(text, pos, ALL);
      const ch = String.fromCodePoint(text.codePointAt(pos)!);
      what = token ? reprText(token.value) : `character ${reprText(ch)}`;
    }
    throw new ParseError(`unexpected ${what} ${where}`);
  }
}

/** A construct the parser has opened and not finished. */
type Frame =
  | { kind: "group" }
  | { kind: "list"; items: ExprNode[] }
  | { kind: "call"; children: ExprNode[]; args: Tree[]; label: Token | null }
  | { kind: "prefix"; level: typeof LEVEL.NOT | typeof LEVEL.UNARY; token: Token }
  // `lower` is null until a slice's colon is read; with none, the subscript is an index.
  | { kind: "subscript"; base: ExprNode; lower: ExprNode | null }
  // A list comprehension after its first `for`: `pending` is the target of a `for` whose iterable,
  // starting at offset `iterStart`, is being read; null while an `if` condition is.
  | { kind: "comp"; elem: ExprNode; clauses: Tree[]; pending: Token | null; iterStart: number }
  // A conditional `then if test else ...`: `test` is null while it is read (the frame acts as a
  // bracket that `else` closes), then the frame waits, like a prefix, for the else branch.
  | { kind: "cond"; then: ExprNode; test: ExprNode | null }
  // Waiting for its next operand, so `operands` is as long as `ops`.
  | { kind: "chain"; level: Level; operands: ExprNode[]; ops: Token[] };

type ChainFrame = Extract<Frame, { kind: "chain" }>;
type CallFrame = Extract<Frame, { kind: "call" }>;
type SubscriptFrame = Extract<Frame, { kind: "subscript" }>;
type CompFrame = Extract<Frame, { kind: "comp" }>;

/** A new tree for an omitted slice bound. */
const omitted = (): Tree => new Tree("omitted", []);

const SLICE_REQUIRED = "inside a slice, `:` ends a bound; write `(a :? b)` for a required value";
const SLICE_STEP = "slices take no step; for a default in a bound, use parentheses: `xs[(n :- 0):]`";
const COMP_ONE_ELEMENT = "a comprehension has one element before `for`; write `[[a, b] for x in xs]`";
const COMP_FOR_ONLY = "`for` makes a list comprehension only: write `[x for x in xs]`";
const COND_NEEDS_ELSE = "a conditional needs `else`: write `a if c else b`";
const COND_IN_TEST = "a conditional's test can't hold another conditional; write `a if (b if c else d) else e`";
const COND_ELSE = "`else` comes after `a if c`, as in `a if c else b`";
const COMP_ELSE = "a comprehension's `if` is a filter; put a conditional there in parentheses: `(a if c else b)`";
const COMP_TARGET = "a comprehension's `for` takes a name, then `in`";
const COMP_ESCAPE = "a comprehension's iterable and `if` stop before `:-`, `:?` and `|`; write `(a :- [])`";
const COMP_RESERVED = (name: string) =>
  `${reprText(name)} can't be a comprehension variable: input, system, item, output and names starting with __ are reserved`;
const COMP_REBOUND = (name: string) => `comprehension variable ${reprText(name)} is already bound here; rename one`;
// The words that are keywords only where an operator is expected.
const CONTEXTUAL_KEYWORDS: ReadonlySet<string> = new Set(["for", "if", "else"]);
// Names a variable would hide from the engine: the scopes every flow reads, and the `__` prefix
// that generated locals (a CASE's `__rN`) use.
const COMP_RESERVED_NAMES: ReadonlySet<string> = new Set(["input", "system", "item", "output"]);
// The escapes sit below `or`, and a comprehension's iterable and condition parse above them.
const ESCAPES: ReadonlySet<string> = new Set(["|", ":-", ":?"]);

/**
 * Read what may follow a slice's colon: the closing `]`, completing `frame` with no upper bound,
 * or else nothing (the upper bound follows).
 *
 * @returns The finished subscript, or null when an upper bound follows.
 */
function afterColon(cursor: Cursor, stack: Frame[], frame: SubscriptFrame): Tree | null {
  // `:?` and `::` lex as `:` here, so name what the author likely meant.
  if (cursor.peek("?")) cursor.fail(cursor.pos, SLICE_REQUIRED);
  if (cursor.peek(":")) cursor.fail(cursor.pos, SLICE_STEP);
  if (!cursor.accept("]")) return null;
  stack.pop();
  return subscriptTree(frame, omitted());
}

/** Complete the clause `frame` is reading, whose iterable or condition is `value` and ends at offset `end`. */
function finishClause(text: string, frame: CompFrame, value: ExprNode, end: number): void {
  frame.clauses.push(
    frame.pending === null
      ? new Tree("comp_if", [value])
      : new Tree("comp_for", [frame.pending, value, new Token("SOURCE", text.slice(frame.iterStart, end).trim())]),
  );
}

/** Read a `for` clause's target and its `in`, and set `frame` to read the iterable next. */
function readTarget(cursor: Cursor, frame: CompFrame): void {
  cursor.skip();
  const name = lexAt(cursor.text, cursor.pos, [NAME]);
  if (name === null) cursor.fail(cursor.pos, COMP_TARGET);
  // `#` and `/` are runtime graph separators, so a variable can't hold them.
  if (EXPRESSION_KEYWORDS.has(name.value) || CONTEXTUAL_KEYWORDS.has(name.value) || /[#/]/.test(name.value)) {
    cursor.fail(name.startPos, COMP_TARGET);
  }
  if (COMP_RESERVED_NAMES.has(name.value) || name.value.startsWith("__")) cursor.fail(name.startPos, COMP_RESERVED(name.value));
  cursor.pos = name.endPos;
  cursor.skip();
  if (lexAt(cursor.text, cursor.pos, [IN]) === null) cursor.fail(name.endPos, COMP_TARGET);
  cursor.next([IN]);
  cursor.skip();
  frame.pending = name;
  frame.iterStart = cursor.pos;
}

/** The precedence level of each binary operator, by token type. */
const OPERATOR_LEVEL: Readonly<Record<string, Level>> = {
  "|": LEVEL.COALESCE,
  ":-": LEVEL.DEFAULT,
  ":?": LEVEL.DEFAULT,
  OR: LEVEL.OR,
  AND: LEVEL.AND,
  COMP_OP: LEVEL.COMPARE,
  IN: LEVEL.COMPARE,
  NOT_IN: LEVEL.COMPARE,
  ADD_OP: LEVEL.SUM,
  MUL_OP: LEVEL.PRODUCT,
  POW_OP: LEVEL.POWER,
};

/** The rule name of an n-ary chain at each level; its operator tokens sit between the operands. */
const CHAIN_RULE: Readonly<Partial<Record<Level, string>>> = {
  [LEVEL.OR]: "or_expr",
  [LEVEL.AND]: "and_expr",
  [LEVEL.SUM]: "sum",
  [LEVEL.PRODUCT]: "product",
  [LEVEL.POWER]: "power",
};

/** The rule name of a default or required expression, by its operator's token type. */
const DEFAULT_RULE: Readonly<Record<string, string>> = {
  ":-": "default_expr",
  ":?": "required_expr",
};

/** The rule name of a comparison, by its operator's token type. */
const COMPARE_RULE: Readonly<Record<string, string>> = {
  COMP_OP: "compare",
  IN: "compare_in",
  NOT_IN: "compare_notin",
};

/** Build the tree of a complete chain, whose `operands` now hold one more than its `ops`. */
function chainTree({ level, operands, ops }: ChainFrame): Tree {
  if (level === LEVEL.COALESCE) return new Tree("coalesce", operands);
  const [first, second] = operands as [ExprNode, ExprNode];
  const op = ops[0]!;
  if (level === LEVEL.DEFAULT) return new Tree(DEFAULT_RULE[op.type]!, [first, second]);
  if (level === LEVEL.COMPARE) return new Tree(COMPARE_RULE[op.type]!, [first, op, second]);
  const children: ExprNode[] = [first];
  ops.forEach((token, i) => children.push(token, operands[i + 1]!));
  return new Tree(CHAIN_RULE[level]!, children);
}

/**
 * Read `.NAME` and `?.NAME` trailers into a refcall's `children`; any keyword is a plain name here.
 * A `?.` trailer is `trailer[NAME, QDOT]`, so `children[0]` stays the NAME for every walker.
 */
function readTrailers(cursor: Cursor, children: ExprNode[]): void {
  for (;;) {
    cursor.skip();
    const start = cursor.pos;
    if (cursor.accept("?.")) {
      const step = cursor.next([NAME]);
      children.push(new Tree("trailer", [step, new Token("QDOT", "?.", start, start + 2)]));
    } else if (cursor.accept(".")) {
      children.push(new Tree("trailer", [cursor.next([NAME])]));
    } else {
      return;
    }
  }
}

/** Close `call`'s argument list, read the trailers after it, and return the refcall. */
function finishCall(cursor: Cursor, call: CallFrame): Tree {
  call.children.push(new Tree("call_suffix", call.args));
  readTrailers(cursor, call.children);
  return new Tree("refcall", call.children);
}

/**
 * Build the tree of a complete subscript: an index `last` when no colon was read, else a slice
 * whose upper bound is `last`.
 */
function subscriptTree(frame: SubscriptFrame, last: ExprNode): Tree {
  const part = frame.lower === null ? new Tree("index", [last]) : new Tree("slice", [frame.lower, last]);
  return new Tree("subscript", [frame.base, part]);
}

/** Add `value`, under the pending label if any, as `call`'s next argument. */
function addArg(call: CallFrame, value: ExprNode): void {
  call.args.push(new Tree("arg", call.label ? [call.label, value] : [value]));
  call.label = null;
}

/**
 * Complete every prefix, chain or conditional frame waiting for its else branch on top of `stack`
 * above `level` around `value`; return the result.
 */
function reduce(stack: Frame[], value: ExprNode, level: Level | 0): ExprNode {
  for (;;) {
    const top = stack.at(-1);
    if (top?.kind === "cond" && top.test !== null && LEVEL.COND > level) {
      stack.pop();
      value = new Tree("if_expr", [top.then, top.test, value]);
      continue;
    }
    if (top?.kind !== "prefix" && top?.kind !== "chain") return value;
    if (top.level <= level) return value;
    stack.pop();
    if (top.kind === "chain") {
      top.operands.push(value);
      value = chainTree(top);
    } else {
      value = top.level === LEVEL.NOT ? new Tree("negate", [top.token, value]) : new Tree("neg", [value]);
    }
  }
}

/**
 * Return the top-most frame on `stack` that opened a bracket: not a pending prefix or chain, nor
 * a conditional waiting for its else branch. A conditional reading its test is one, closed by
 * `else`.
 */
function nearestBracket(stack: Frame[]): Frame | undefined {
  return stack.findLast(
    (frame) => frame.kind !== "prefix" && frame.kind !== "chain" && !(frame.kind === "cond" && frame.test !== null),
  );
}

/** Whether `frame` is a conditional reading its test, which only `else` may end. */
function readsTest(frame: Frame | undefined): boolean {
  return frame?.kind === "cond" && frame.test === null;
}

/**
 * Parse `text` into a tree and scope its comprehension variables (`scopeComprehensions`), or throw
 * a `ParseError`; no other post-parse checks.
 */
function parseRaw(text: string): ExprNode {
  // Typed, so a `cursor.fail(...)` (which never returns) narrows the frame checked before it.
  const cursor: Cursor = new Cursor(text);
  const stack: Frame[] = [];
  // The operand just read; null while one is expected.
  let value: ExprNode | null = null;
  for (;;) {
    if (value === null) {
      const top = stack.at(-1);
      const offersNot = !(
        (top?.kind === "prefix" && top.level === LEVEL.UNARY) ||
        (top?.kind === "chain" && top.level >= LEVEL.COMPARE)
      );
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
          // A literal: NUMBER, STRING, BOOL, NULL or INF.
          value = token;
      }
      continue;
    }
    if (cursor.atEnd()) {
      value = reduce(stack, value, 0);
      if (stack.length === 0) return scopeComprehensions(value, cursor);
      cursor.fail(text.length, readsTest(stack.at(-1)) ? COND_NEEDS_ELSE : undefined);
    }
    // Scan for the nearest bracket only when a colon is next: a scan per operator would make a
    // long chain of pending prefixes quadratic.
    const inSlice = cursor.peek(":") && nearestBracket(stack)?.kind === "subscript";
    const token = cursor.next(inSlice ? OPERATOR_IN_SLICE : OPERATOR);
    if (token.type === "FOR") {
      value = reduce(stack, value, 0);
      let frame = stack.at(-1);
      if (frame?.kind === "list" && frame.items.length === 0) {
        stack.pop();
        frame = { kind: "comp", elem: value, clauses: [], pending: null, iterStart: 0 };
        stack.push(frame);
      } else if (frame?.kind === "comp") {
        finishClause(text, frame, value, token.startPos);
      } else {
        const why = frame?.kind === "list" ? COMP_ONE_ELEMENT : readsTest(frame) ? COND_NEEDS_ELSE : COMP_FOR_ONLY;
        cursor.fail(token.startPos, why);
      }
      readTarget(cursor, frame);
      value = null;
      continue;
    }
    if (token.type === "IF") {
      // A conditional waiting for its else branch stays: `a if b else c if d else e` nests to the right.
      value = reduce(stack, value, LEVEL.COND);
      const frame = stack.at(-1);
      if (frame?.kind === "comp") {
        finishClause(text, frame, value, token.startPos);
        frame.pending = null;
      } else if (readsTest(frame)) {
        cursor.fail(token.startPos, COND_IN_TEST);
      } else {
        stack.push({ kind: "cond", then: value, test: null });
      }
      value = null;
      continue;
    }
    if (token.type === "ELSE") {
      value = reduce(stack, value, LEVEL.COND);
      const frame = stack.at(-1);
      if (!readsTest(frame)) cursor.fail(token.startPos, frame?.kind === "comp" ? COMP_ELSE : COND_ELSE);
      (frame as Extract<Frame, { kind: "cond" }>).test = value;
      value = null;
      continue;
    }
    if (token.type === "[") {
      // Before any reduce, so the subscript binds to the bare operand, tighter than `-` and `**`.
      const frame: SubscriptFrame = { kind: "subscript", base: value, lower: null };
      stack.push(frame);
      value = null;
      if (cursor.accept(":")) {
        frame.lower = omitted();
        value = afterColon(cursor, stack, frame);
      }
      continue;
    }
    if (token.type === ":") {
      value = reduce(stack, value, 0);
      const frame = stack.at(-1) as SubscriptFrame;
      if (frame.lower !== null) cursor.fail(token.startPos, SLICE_STEP);
      frame.lower = value;
      value = afterColon(cursor, stack, frame);
      continue;
    }
    if (ESCAPES.has(token.type) && nearestBracket(stack)?.kind === "comp") cursor.fail(token.startPos, COMP_ESCAPE);
    const level = OPERATOR_LEVEL[token.type];
    if (level !== undefined) {
      value = reduce(stack, value, level);
      const top = stack.at(-1);
      if (top?.kind === "chain" && top.level === level) {
        // `a :- b :- c` and `a == b in c` don't parse: those levels take two operands.
        if (level === LEVEL.DEFAULT || level === LEVEL.COMPARE) cursor.fail(token.startPos);
        top.operands.push(value);
        top.ops.push(token);
      } else {
        stack.push({ kind: "chain", level, operands: [value], ops: [token] });
      }
      value = null;
      continue;
    }
    value = reduce(stack, value, 0);
    const frame = stack.at(-1);
    if (token.type === ")" && frame?.kind === "group") {
      stack.pop();
    } else if (token.type === ")" && frame?.kind === "call") {
      stack.pop();
      addArg(frame, value);
      value = finishCall(cursor, frame);
    } else if (token.type === "]" && frame?.kind === "subscript") {
      stack.pop();
      value = subscriptTree(frame, value);
    } else if (token.type === "]" && frame?.kind === "comp") {
      finishClause(text, frame, value, token.startPos);
      stack.pop();
      value = new Tree("list_comp", [frame.elem, ...frame.clauses]);
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
      cursor.fail(token.startPos, readsTest(frame) ? COND_NEEDS_ELSE : undefined);
    }
  }
}

// Keyed by the read's head token, not its `comp_var` Tree: `masked` and `falsified` in
// compose/helpers/condition_guards.ts copy every Tree of a condition but keep its Tokens.
/** Each comprehension variable read's head NAME token, to its `for` target token. */
const COMP_VAR_TARGETS = new WeakMap<Token, Token>();

/**
 * Return the key of a comprehension variable read's binding occurrence, `<name>@<offset of its
 * for target>`, or `null` for a `comp_var` the parser didn't scope. `@` is no NAME character, so a
 * key never equals a reference path. Copies of the read's Tree that keep its head Token keep the
 * key.
 */
export function compVarKey(node: Tree): string | null {
  const head = node.children[0];
  const target = head instanceof Token ? COMP_VAR_TARGETS.get(head) : undefined;
  return target === undefined ? null : bindingKey(target);
}

/** Return the key of a comprehension's `for` target token, the key `compVarKey` gives its reads. */
export function bindingKey(target: Token): string {
  return `${target.value}@${target.startPos}`;
}

/**
 * Turn each read of a comprehension variable in `root` into `comp_var[NAME, trailer*]`, in place,
 * and return `root`.
 *
 * A variable is in scope in its comprehension's elem and in every clause after its own `for`,
 * nested comprehensions included, but not in its own or an earlier iterable: the first iterable
 * runs in the enclosing scope. Only a reference with no call suffix is a read, and
 * `isinstance`'s type name is not one. So every walker that matches `refcall` skips variables
 * with no change. Iterative, as the parser is.
 *
 * @throws {ParseError} If a `for` rebinds a name already bound there.
 */
function scopeComprehensions(root: ExprNode, cursor: Cursor): ExprNode {
  if (!(root instanceof Tree)) return root;
  // A scope maps each variable name to its `for` target token.
  const stack: [Tree, ReadonlyMap<string, Token>][] = [[root, new Map()]];
  const visit = (parent: Tree, i: number, scope: ReadonlyMap<string, Token>): void => {
    const child = parent.children[i];
    if (!(child instanceof Tree)) return;
    const target = child.data === "refcall" ? scope.get(String(child.children[0])) : undefined;
    if (target !== undefined && callSuffixOf(child) === undefined) {
      COMP_VAR_TARGETS.set(child.children[0] as Token, target);
      parent.children[i] = new Tree("comp_var", child.children);
    } else {
      stack.push([child, scope]);
    }
  };
  while (stack.length > 0) {
    const [node, scope] = stack.pop()!;
    if (node.data === "list_comp") {
      let bound = scope;
      for (const clause of node.children.slice(1) as Tree[]) {
        if (clause.data === "comp_for") {
          const target = clause.children[0] as Token;
          if (bound.has(target.value)) cursor.fail(target.startPos, COMP_REBOUND(target.value));
          visit(clause, 1, bound);
          bound = new Map([...bound, [target.value, target]]);
        } else {
          visit(clause, 0, bound);
        }
      }
      visit(node, 0, bound);
      continue;
    }
    const callSuffix = node.data === "refcall" ? callSuffixOf(node) : undefined;
    if (callSuffix !== undefined) {
      // An `arg` is never a reference itself, so it is walked rather than visited.
      for (const arg of valueArgs(String(node.children[0]), callSuffix)) stack.push([arg, scope]);
      continue;
    }
    for (let i = 0; i < node.children.length; i++) visit(node, i, scope);
  }
  return root;
}

const RESERVED_REFERENCE_ROOTS: ReadonlySet<string> = new Set(["and", "or", "not", "in"]);

/** The keywords of the grammar; none of them can be a reference root. */
export const EXPRESSION_KEYWORDS: ReadonlySet<string> = new Set([
  ...RESERVED_REFERENCE_ROOTS,
  "true",
  "false",
  "null",
  "none",
  "inf",
]);

/**
 * Return why no expression can call a function named `name`, else `null`.
 *
 * A callee is one `NAME` token lexed where an operand starts, so `name` must lex as exactly one
 * there: a literal word (`true`, `inf`) lexes as a literal, even before a `#` or `/`
 * (`true/x(1)` reads as `true / x(1)`). `isinstance(...)` is always the type test. The operator
 * words stay callable: `and(x)` is a call, and so is `not(x)` after an operator (`1 + not(x)`).
 */
export function uncallableReason(name: string): string | null {
  const token = lexAt(name, 0, OPERAND_WITHOUT_NOT);
  if (token !== null && (token.type === "BOOL" || token.type === "NULL" || token.type === "INF")) {
    return token.endPos === name.length
      ? `\`${name}\` is a literal`
      : `it starts with the literal \`${token.value}\``;
  }
  if (token?.type !== "NAME" || token.endPos !== name.length) {
    return "it is no name: a call names a letter or `_` followed by letters, digits and `_`";
  }
  if (name === ISINSTANCE) return "`isinstance(x, T)` is the type test";
  return null;
}

/** Return the `call_suffix` of the refcall `node`, or undefined when it is a reference. */
export function callSuffixOf(node: Tree): Tree | undefined {
  return node.children.find((child): child is Tree => child instanceof Tree && child.data === "call_suffix");
}

/**
 * Return the `arg` subtrees of a call whose values are expressions.
 *
 * Every argument of a builtin is an expression; `isinstance(x, T)` has only `x`.
 */
export function valueArgs(callee: string, callSuffix: Tree): Tree[] {
  const args = callSuffix.children as Tree[];
  return callee === ISINSTANCE ? args.slice(0, 1) : args;
}

/**
 * Return the index in `text` of the `}` closing a `${...}` span whose interior begins at `start`
 * (brace depth and quote aware), or null if the span is unbalanced.
 */
export function findSpanEnd(text: string, start: number): number | null {
  let depth = 1;
  let quote: string | null = null;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (quote !== null) {
      if (c === quote) quote = null;
    } else if (c === "'" || c === '"') {
      quote = c;
    } else if (c === "{") {
      depth += 1;
    } else if (c === "}") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return null;
}

/**
 * Return the interior of `text` when the trimmed text is exactly one `${...}` span, else null.
 *
 * Quote aware, so `${x == "}"}` is one span and `${a} > ${b}` is not.
 */
export function wholeSpanInterior(text: string): string | null {
  text = text.trim();
  if (!text.startsWith("${")) return null;
  const end = findSpanEnd(text, 2);
  return end === text.length - 1 ? text.slice(2, end) : null;
}

// A character that would fuse with an unwrapped name: `not${x}` must become `not x`.
const NAME_CHAR = /^[A-Za-z0-9_#/]/;

/**
 * Return whether `text` parses within the nesting limit with no reserved word as a reference root
 * (a rewrite worth suggesting). `parseExpr`'s other checks report their own errors once the spans
 * are gone.
 */
function parses(text: string): boolean {
  try {
    const tree = parseRaw(text);
    rejectDeepExpression(tree);
    rejectReservedReferenceRoots(tree);
  } catch (error) {
    if (error instanceof ParseError || error instanceof ExpressionError) return false;
    throw error;
  }
  return true;
}

/**
 * Return whether `text` parses as one reference or one literal (no operator or call). A reserved
 * word (`in`, `and`, ...) is not one, so unwrapping `${in}` can't turn it into an operator.
 */
function isLoneAtom(text: string): boolean {
  let tree: ExprNode;
  try {
    tree = parseRaw(text);
  } catch (error) {
    if (error instanceof ParseError) return false;
    throw error;
  }
  if (tree instanceof Token) return true;
  if (tree.data !== "refcall" || callSuffixOf(tree) !== undefined) return false;
  const head = tree.children[0];
  return !(head instanceof Token && head.type === "NAME" && RESERVED_REFERENCE_ROOTS.has(head.value));
}

/**
 * Return `text` with each `${X}` outside a string literal replaced by `X`, or `(X)` when `X` is
 * more than one reference or literal; null when a `${` is unbalanced or spans nest deeper than
 * `limit`. `depth` counts the spans around `text`.
 */
function unwrapSpans(text: string, depth = 0, limit = current().maxValueNesting): string | null {
  if (depth > limit) return null;
  const out: string[] = [];
  let quote: string | null = null;
  let i = 0;
  while (i < text.length) {
    const c = text[i]!;
    if (quote !== null) {
      out.push(c);
      if (c === quote) quote = null;
      i += 1;
    } else if (c === "'" || c === '"') {
      quote = c;
      out.push(c);
      i += 1;
    } else if (c === "$" && text[i + 1] === "{") {
      const end = findSpanEnd(text, i + 2);
      if (end === null) return null;
      const unwrapped = unwrapSpans(text.slice(i + 2, end), depth + 1, limit);
      if (unwrapped === null) return null;
      const inner = unwrapped.trim();
      let piece = isLoneAtom(inner) ? inner : `(${inner})`;
      // `out` holds single characters and whole pieces; a piece can start with a space (` x`), so
      // the last character of the last entry decides whether the names would run together.
      if (out.length > 0 && NAME_CHAR.test(out.at(-1)!.at(-1)!)) piece = " " + piece;
      if (NAME_CHAR.test(text.slice(end + 1, end + 2))) piece += " ";
      out.push(piece);
      i = end + 1;
    } else {
      out.push(c);
      i += 1;
    }
  }
  return out.join("");
}

/** Return whether `text` has a `${` outside a string literal. */
function hasSpan(text: string): boolean {
  let quote: string | null = null;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quote !== null) {
      if (c === quote) quote = null;
    } else if (c === "'" || c === '"') {
      quote = c;
    } else if (c === "$" && text[i + 1] === "{") {
      return true;
    }
  }
  return false;
}

/**
 * Throw if the expression `text` holds a `${...}`, showing the rewrite.
 *
 * `${...}` marks an expression inside a YAML string, so inside an expression values are named
 * directly and grouped with parentheses. A `${...}` inside a quoted string literal is text and
 * allowed.
 *
 * @param text One expression: a span interior or a bare condition.
 * @param options.inSpan `text` is the interior of a template span, so the rewrite is shown as
 *   the whole fixed span (`${a :- b}`) rather than the bare expression (`a :- b`).
 * @throws {ExpressionError} If `text` holds a `${...}` outside a string literal. The message
 *   shows the rewrite when the unwrapped text parses.
 */
export function rejectNestedSpans(text: string, { inSpan = false }: { inSpan?: boolean } = {}): void {
  if (!hasSpan(text)) return;
  let message = "`${...}` can't appear inside an expression";
  let rewrite = unwrapSpans(text);
  if (rewrite !== null && parses(rewrite)) {
    rewrite = rewrite.trim();
    message += `; write \`${inSpan ? "${" + rewrite + "}" : rewrite}\``;
  }
  throw new ExpressionError(message);
}

// Parse nodes that only group the parts of a reference or call; they add no level.
const WRAPPER_NODES: ReadonlySet<string> = new Set([
  "trailer",
  "call_suffix",
  "arg",
  "index",
  "slice",
  "omitted",
  "comp_for",
  "comp_if",
  "comp_var",
]);

/**
 * Throw if a parsed expression nests deeper than the value nesting limit.
 *
 * An expression nests no deeper than a value, so one limit, `current().maxValueNesting`, bounds
 * both. Each operator chain (`sum`, `product`, `power`, `and_expr`, `or_expr`), unary operator,
 * comparison, escape (`|`, `:-`, `:?`), list literal, call and subscript is one level; a plain
 * reference and the parts of a reference, call or slice add none, and parentheses make no node.
 *
 * @throws {ExpressionError} If any path from the root passes more than the limit.
 */
function rejectDeepExpression(tree: ExprNode): void {
  const limit = current().maxValueNesting;
  const stack: [ExprNode, number][] = [[tree, 0]];
  while (stack.length > 0) {
    const [node, outer] = stack.pop()!;
    if (!(node instanceof Tree)) continue;
    let depth = outer;
    if (!WRAPPER_NODES.has(node.data) && (node.data !== "refcall" || callSuffixOf(node))) {
      depth += 1;
      if (depth > limit) {
        throw new ExpressionError(
          `expression nests deeper than ${limit} levels, the limit for a yian expression (each call, ` +
            "list, unary operator, comparison, and chain of operators is a level); split it into smaller " +
            "bindings or steps",
        );
      }
    }
    for (const child of node.children) stack.push([child, depth]);
  }
}

/**
 * Throw if a parsed expression holds a float literal outside the double range.
 *
 * The walk pops the last child first, so of two such literals the rightmost is reported.
 *
 * @throws {ExpressionError} If a `NUMBER` token with a `.` is not a finite double; the message
 *   gives the count of integer digits after any leading zeros, never the digits.
 */
function rejectOverflowingFloatLiterals(tree: ExprNode): void {
  const stack: ExprNode[] = [tree];
  while (stack.length > 0) {
    const node = stack.pop()!;
    if (node instanceof Tree) {
      for (const child of node.children) stack.push(child);
    } else if (node.type === "NUMBER" && node.value.includes(".") && !Number.isFinite(Number(node.value))) {
      const digits = node.value.slice(0, node.value.indexOf(".")).replace(/^0+/, "").length;
      throw new ExpressionError(
        `float literal with ${digits} digits before the decimal point is out of range; write inf for infinity`,
      );
    }
  }
}

/**
 * Throw if a `refcall` is a call with a `trailer` before its `call_suffix`.
 *
 * Builtins are bare-callee only, matching prompt syntax: `a.b(x)` is rejected, while trailers
 * after the parentheses (`fn(x).field`) are dotted access on the result and fine.
 */
function rejectDottedCallee(tree: ExprNode): void {
  if (!(tree instanceof Tree)) return;
  for (const node of iterSubtrees(tree)) {
    if (node.data !== "refcall") continue;
    let hasCall = false;
    let trailerBeforeCall = false;
    for (const child of node.children) {
      if (child instanceof Tree && child.data === "call_suffix") hasCall = true;
      else if (child instanceof Tree && child.data === "trailer" && !hasCall) trailerBeforeCall = true;
    }
    if (hasCall && trailerBeforeCall) {
      throw new ExpressionError(
        "builtin call must have a bare callee (no dotted access before '('): a form like 'a.b(x)' is not allowed",
      );
    }
  }
}

/** Return whether `node` is a `refcall` that is one non-keyword `NAME` with no trailers or call suffix. */
export function isBareName(node: ExprNode): node is Tree & { readonly data: "refcall" } {
  if (!(node instanceof Tree) || node.data !== "refcall" || node.children.length !== 1) return false;
  const head = node.children[0];
  return head instanceof Token && head.type === "NAME" && !RESERVED_REFERENCE_ROOTS.has(head.value);
}

/**
 * Require every `isinstance(...)` call to be `isinstance(<expr>, <TypeName>)`.
 *
 * The second argument names a type, so it must be a bare `NAME`: no dotted path, call, or
 * literal. Keyword arguments and dotted access on the boolean result (`isinstance(x, T).f`) are
 * not accepted.
 */
function validateIsinstance(tree: ExprNode): void {
  if (!(tree instanceof Tree)) return;
  for (const node of iterSubtrees(tree)) {
    if (node.data !== "refcall" || String(node.children[0]) !== ISINSTANCE) continue;
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) continue;
    if (node.children.at(-1) !== callSuffix) {
      throw new ExpressionError("isinstance(...) gives a bool; it has no fields to read");
    }
    const args = callSuffix.children as Tree[];
    if (args.some((arg) => arg.children.length === 2)) {
      throw new ExpressionError("isinstance takes positional arguments, not keyword arguments");
    }
    if (args.length !== 2) {
      throw new ExpressionError(
        `isinstance takes exactly two arguments (a value and a type name), got ${args.length}`,
      );
    }
    if (!isBareName(args[1]!.children[0]!)) {
      throw new ExpressionError("isinstance's second argument must be a bare type name, e.g. isinstance(x, Buy)");
    }
  }
}

/** Throw if a call repeats a keyword argument, which no callee accepts. */
function validateCallArguments(tree: ExprNode): void {
  if (!(tree instanceof Tree)) return;
  for (const node of iterSubtrees(tree)) {
    if (node.data !== "refcall") continue;
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) continue;
    const seen = new Set<string>();
    for (const arg of callSuffix.children as Tree[]) {
      if (arg.children.length !== 2) continue;
      const keyword = String(arg.children[0]);
      if (seen.has(keyword)) {
        throw new ExpressionError(`${String(node.children[0])}(...) repeats the keyword ${reprText(keyword)}`);
      }
      seen.add(keyword);
    }
  }
}

/**
 * Throw if an operator word parsed contextually as a reference root.
 *
 * A `refcall` with a `call_suffix` is a builtin call rather than a reference, so its callee is
 * outside this rule. Trailer segments and keyword-argument labels are likewise not roots.
 */
function rejectReservedReferenceRoots(tree: ExprNode): void {
  if (!(tree instanceof Tree)) return;
  for (const node of iterSubtrees(tree)) {
    if (node.data !== "refcall" || callSuffixOf(node)) continue;
    const head = node.children[0];
    if (head instanceof Token && head.type === "NAME" && RESERVED_REFERENCE_ROOTS.has(head.value)) {
      throw new ExpressionError(`reserved expression keyword ${reprText(head.value)} cannot be used as a reference root`);
    }
  }
}

/**
 * Parse one unified `${...}` expression into a tree.
 *
 * This does no evaluation; it is the shared front end for the evaluator in `expressions.ts`.
 *
 * @returns The parse tree, shaped as the module's "Tree shape" section says: a `refcall` for a
 *   reference or call, `coalesce` (only when a `|` appears) / `default_expr` / `required_expr` at
 *   the top, and the boolean, comparison and arithmetic nodes below. A lone literal (a `NUMBER`,
 *   `STRING`, `BOOL`, `NULL` or `INF`) comes back as a bare `Token`, since a level with one
 *   operand makes no node.
 * @throws {ExpressionError} If `text` holds a `${...}` (see `rejectNestedSpans`), does not parse
 *   (`could not parse expression '<text>': unexpected <what> at line L, column C`, or a slice's
 *   step or `:?`, or a misplaced `for`/`if`/`else`, a bad comprehension target or an escape in
 *   an iterable or condition, with the fix in place of `unexpected <what>`), nests deeper
 *   than `current().maxValueNesting` levels, holds a float literal outside the double range, is a
 *   dotted-callee builtin call, calls `isinstance` with anything but a value and a bare type
 *   name, repeats a keyword argument, or uses a reserved operator word as a reference root.
 */
export function parseExpr(text: string): ExprNode {
  rejectNestedSpans(text);
  let tree: ExprNode;
  try {
    tree = parseRaw(text);
  } catch (error) {
    if (!(error instanceof ParseError)) throw error;
    throw new ExpressionError(`could not parse expression ${reprText(text)}: ${error.message}`, {
      cause: error,
    });
  }
  // First: a too-deep expression reports the nesting limit before any other
  // post-parse check. (The comprehension scoping checks run inside the parse, without recursion.)
  rejectDeepExpression(tree);
  rejectOverflowingFloatLiterals(tree);
  rejectDottedCallee(tree);
  // Before the reserved-root check, so `isinstance(x, in)` reports the type-name rule.
  validateIsinstance(tree);
  validateCallArguments(tree);
  rejectReservedReferenceRoots(tree);
  return tree;
}
