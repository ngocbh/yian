/**
 * Parse tests for the unified `${...}` expression grammar (`expr/grammar.ts`): references,
 * operators and their precedence, builtin calls, keyword arguments, list literals, groups,
 * coalesce/default/required, literals, token positions, parse-error messages, the error classes,
 * `iterSubtrees` and `iterSubtreesTopdown` order, and the checks after the parse: reserved
 * reference roots, the dotted callee, the nesting limit, float overflow and the order the checks
 * run in; and the `${...}` helpers: a nested span's error and rewrite, `findSpanEnd` and
 * `wholeSpanInterior`; and the `scanTemplate` halves of the reserved-keyword tests.
 *
 * Also: subscripts (list slices), their tree, how tightly they bind, and the colon rule inside a
 * subscript's brackets; list comprehensions, their tree and clause order, and the scoping pass
 * that turns comprehension-variable reads into `comp_var`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
import { ConstructorTypeError, ExpressionError, RequiredError } from "../../src/expr/expressions.ts";
import {
  compVarKey,
  EXPRESSION_KEYWORDS,
  type ExprNode,
  findSpanEnd,
  isBareName,
  iterSubtrees,
  iterSubtreesTopdown,
  parseExpr,
  Token,
  Tree,
  wholeSpanInterior,
} from "../../src/expr/grammar.ts";
import { scanTemplate } from "../../src/expr/template.ts";

/** A position-blind form of `node`, as Lark's `Tree`/`Token` equality ignores positions. */
type Plain = { type: string; value: string } | { data: string; children: Plain[] };

function plain(node: ExprNode): Plain {
  if (node instanceof Token) return { type: node.type, value: node.value };
  return { data: node.data, children: node.children.map(plain) };
}

/** Parse `text` and return its position-blind form. */
function parsed(text: string): Plain {
  return plain(parseExpr(text));
}

/** Parse `text` and require a `Tree`. */
function tree(text: string): Tree {
  const result = parseExpr(text);
  if (!(result instanceof Tree)) throw new Error(`${text} did not parse to a Tree`);
  return result;
}

/** The set of rule names anywhere in `node`, the coarse shape probe. */
function kinds(node: Tree): Set<string> {
  return new Set(iterSubtrees(node).map((n) => n.data));
}

const name = (value: string) => new Token("NAME", value);
const ref = (...parts: string[]) =>
  new Tree("refcall", [name(parts[0]!), ...parts.slice(1).map((p) => new Tree("trailer", [name(p)]))]);
const num = (value: string) => new Token("NUMBER", value);
const arg = (...children: ExprNode[]) => new Tree("arg", children);
const call = (callee: string, args: Tree[], ...after: string[]) =>
  new Tree("refcall", [
    name(callee),
    new Tree("call_suffix", args),
    ...after.map((p) => new Tree("trailer", [name(p)])),
  ]);

const not = new Token("NOT", "not");
const op = (type: string, value: string) => new Token(type, value);

/** Expect `parseExpr(text)` to fail with a parse error. */
function expectParseError(text: string): void {
  expect(() => parseExpr(text)).toThrow(ExpressionError);
  expect(() => parseExpr(text)).toThrow("could not parse expression");
}

/** Expect `parseExpr(text)` to fail with exactly the parse error `detail`. */
function expectParseErrorDetail(text: string, detail: string): void {
  let error: unknown;
  try {
    parseExpr(text);
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  const quoted = `'${text.replaceAll("\n", "\\n").replaceAll("\r", "\\r")}'`;
  expect((error as Error).message).toBe(`could not parse expression ${quoted}: ${detail}`);
}

/** Expect `parseExpr(text)` to fail with exactly `message`. */
function expectMessage(text: string, message: string): void {
  let error: unknown;
  try {
    parseExpr(text);
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe(message);
}

const reservedRoot = (keyword: string) => `reserved expression keyword '${keyword}' cannot be used as a reference root`;
const nestedSpan = (rewrite: string | null) =>
  "`${...}` can't appear inside an expression" + (rewrite === null ? "" : `; write \`${rewrite}\``);
const DOTTED_CALLEE =
  "builtin call must have a bare callee (no dotted access before '('): a form like 'a.b(x)' is not allowed";

describe("references", () => {
  test("bare ref", () => {
    expect(kinds(tree("a")).has("refcall")).toBe(true);
    expect(parsed("a")).toEqual(plain(ref("a")));
  });

  test("dotted ref is one refcall", () => {
    const t = tree("a.b.c");
    expect(iterSubtrees(t).filter((n) => n.data === "refcall")).toHaveLength(1);
    expect(plain(t)).toEqual(plain(ref("a", "b", "c")));
  });

  test("wrapped ref is rejected", () => {
    // `${...}` marks an expression in a YAML string; inside an expression a name is bare.
    expectMessage("${a}", nestedSpan("a"));
  });

  test("ref with hash, slash and item", () => {
    expect(parsed("node#0.output")).toEqual(plain(ref("node#0", "output")));
    expect(parsed("def/child.output")).toEqual(plain(ref("def/child", "output")));
    expect(parsed("item.title")).toEqual(plain(ref("item", "title")));
  });

  test("whitespace around a dot", () => {
    expect(parsed("a . b")).toEqual(plain(ref("a", "b")));
  });

  test("a slash belongs to the name", () => {
    expect(parsed("a/b")).toEqual(plain(ref("a/b")));
    expectParseErrorDetail("a/ b", "unexpected 'b' at line 1, column 4");
  });

  test("a lone expression is not wrapped", () => {
    expect(tree("a").data).toBe("refcall");
    expect(tree("1 + 2").data).toBe("sum");
  });

  test("not followed by parentheses is still the operator", () => {
    expect(tree("not(x)").data).toBe("negate");
  });

  test("reserved keyword cannot be a bare reference root", () => {
    for (const keyword of ["and", "or", "in"]) expectMessage(keyword, reservedRoot(keyword));
    // At the start `not` lexes as the operator, which then lacks its operand.
    expectParseErrorDetail("not", "unexpected end of expression at line 1, column 4");
  });

  test("reserved keyword cannot be an operand reference root", () => {
    const cases: [string, string][] = [
      ["x and and", "and"],
      ["x or or", "or"],
      ["x == not", "not"],
      ["x in in", "in"],
    ];
    for (const [text, keyword] of cases) expectMessage(text, reservedRoot(keyword));
  });

  test("reserved keyword cannot be the root of a dotted reference", () => {
    expectMessage("and.x", reservedRoot("and"));
    expectMessage("x + not.y", reservedRoot("not"));
  });

  test("wrapped reserved keyword is a nested-span error", () => {
    for (const keyword of ["and", "or", "not", "in"]) {
      for (const prefix of ["", "x :- "]) {
        // A reserved word can't be a reference, so no rewrite is offered.
        expectMessage(`${prefix}\${${keyword}}`, nestedSpan(null));
      }
    }
  });

  test("a not as a keyword argument's whole value is a parse error", () => {
    expectParseError("f(a=not)");
  });

  test("keyword prefixes stay legal in generated roots", () => {
    for (const root of ["and/child", "or#0", "not/child", "not#0", "in#0"]) {
      expect(parsed(`${root}.output`)).toEqual(plain(ref(root, "output")));
      scanTemplate(`\${${root}.output}`);
    }
  });

  test("keywords are legal dotted fields", () => {
    for (const field of ["and", "or", "not", "in", "true", "inf"]) {
      expect(parsed(`record.output.${field}`)).toEqual(plain(ref("record", "output", field)));
      scanTemplate(`\${record.output.${field}}`);
    }
  });

  const QDOT = () => new Token("QDOT", "?.");
  /** A refcall whose steps are `[name, optional]`. */
  const qref = (head: string, ...steps: [string, boolean][]) =>
    new Tree("refcall", [
      name(head),
      ...steps.map(([step, optional]) => new Tree("trailer", optional ? [name(step), QDOT()] : [name(step)])),
    ]);

  test("?. is a trailer whose NAME comes first, then a QDOT", () => {
    expect(parsed("a.output?.k.m")).toEqual(plain(qref("a", ["output", false], ["k", true], ["m", false])));
    expect(parsed("a ?. b")).toEqual(plain(qref("a", ["b", true])));
    expect(parsed("f(x)?.k")).toEqual(
      plain(new Tree("refcall", [name("f"), new Tree("call_suffix", [arg(ref("x"))]), new Tree("trailer", [name("k"), QDOT()])])),
    );
  });

  test("?. leaves :? and comprehension variables alone", () => {
    expect(kinds(tree('a :? "m"')).has("required_expr")).toBe(true);
    const comp = tree("[m?.t for m in ms]");
    expect(plain(comp.children[0]!)).toEqual(plain(new Tree("comp_var", [name("m"), new Tree("trailer", [name("t"), QDOT()])])));
  });

  test("?. before a call is a dotted callee, and ?. needs a name", () => {
    expectMessage("a?.b(x)", DOTTED_CALLEE);
    expectParseError("a?.");
    expectParseError("a?.1");
  });
});

describe("builtin calls", () => {
  test("pure builtin call", () => {
    expect(parsed("upper(name)")).toEqual(plain(call("upper", [arg(ref("name"))])));
  });

  test("call with a string argument", () => {
    const t = tree('join(items, ", ")');
    expect(kinds(t).has("call_suffix")).toBe(true);
    expect(plain(t)).toEqual(plain(call("join", [arg(ref("items")), arg(new Token("STRING", '", "'))])));
  });

  test("dotted access on a call result", () => {
    expect(parsed("fn(x).field")).toEqual(plain(call("fn", [arg(ref("x"))], "field")));
  });

  test("dotted callee rejected", () => {
    expectMessage("a.b(x)", DOTTED_CALLEE);
  });

  test("empty argument call", () => {
    const t = tree("fn()");
    expect(kinds(t).has("call_suffix")).toBe(true);
    expect(kinds(t).has("arg")).toBe(false);
  });

  test("kwarg distinguishable from positional", () => {
    const kwArg = iterSubtrees(tree("foo(a=1)")).find((n) => n.data === "arg")!;
    const posArg = iterSubtrees(tree("foo(1)")).find((n) => n.data === "arg")!;
    expect(kwArg.children).toHaveLength(2);
    expect(plain(kwArg.children[0]!)).toEqual({ type: "NAME", value: "a" });
    expect(posArg.children).toHaveLength(1);
    expect(plain(posArg.children[0]!)).toEqual({ type: "NUMBER", value: "1" });
  });

  test("a keyword argument's value may be a not", () => {
    const expected = plain(call("f", [arg(name("a"), new Tree("negate", [not, ref("x")]))]));
    for (const text of ["f(a=not x)", "f(a = not x)", "f(a=not(x))"]) expect(parsed(text)).toEqual(expected);
  });

  test("a comparison argument is positional", () => {
    expect(parsed("f(a == b)")).toEqual(plain(call("f", [arg(new Tree("compare", [ref("a"), op("COMP_OP", "=="), ref("b")]))])));
    expectParseError("f(a===b)");
  });

  test("keyword argument shapes", () => {
    expect(parsed("f(a = 1)")).toEqual(plain(call("f", [arg(name("a"), num("1"))])));
    expect(parsed("f(y=1, x)")).toEqual(plain(call("f", [arg(name("y"), num("1")), arg(ref("x"))])));
    expect(parsed("f(x=1).y")).toEqual(plain(call("f", [arg(name("x"), num("1"))], "y")));
    expect(parsed("f().x")).toEqual(plain(call("f", [], "x")));
  });

  test("whitespace before the parentheses still makes a call", () => {
    expect(parsed("a (b)")).toEqual(plain(call("a", [arg(ref("b"))])));
    expect(parsed("f (x) . y")).toEqual(plain(call("f", [arg(ref("x"))], "y")));
    expect(parsed("f(x)\n.y")).toEqual(plain(call("f", [arg(ref("x"))], "y")));
  });

  test("keyword-shaped callees are calls", () => {
    for (const callee of ["and", "or", "in"]) {
      expect(parsed(`${callee}(x)`)).toEqual(plain(call(callee, [arg(ref("x"))])));
    }
  });

  test("keyword-shaped labels are labels", () => {
    for (const label of ["and", "or", "in"]) {
      expect(parsed(`fn(${label}=x)`)).toEqual(plain(call("fn", [arg(name(label), ref("x"))])));
    }
    expect(parsed("f(not#0=1)")).toEqual(plain(call("f", [arg(name("not#0"), num("1"))])));
  });

  test("labels that lex as keywords, or aren't names, are parse errors", () => {
    for (const text of ["f(true=1)", "fn(not=x)", "f(inf=1)", "f(a = = b)", "f(a.b=1)"]) {
      expectParseError(text);
    }
  });
});

describe("lists and groups", () => {
  test("list literals", () => {
    expect(parsed('[1, "x"]')).toEqual(plain(new Tree("list_lit", [num("1"), new Token("STRING", '"x"')])));
    expect(parsed("[]")).toEqual(plain(new Tree("list_lit", [])));
    expect(parsed("[ ]")).toEqual(plain(new Tree("list_lit", [])));
  });

  test("parentheses make no node", () => {
    expect(parsed("((x))")).toEqual(plain(ref("x")));
  });

  test("malformed lists, calls and groups", () => {
    for (const text of ["[1,]", "f(1,)", "f(,)", "()", "(a, b)"]) expectParseError(text);
  });

  test("no trailer after a group, and no call of a call", () => {
    for (const text of ["(a).b", "f(x)(y)", "f(a=1)(2)", "f(x).g(y)"]) expectParseError(text);
  });
});

describe("arithmetic, comparisons and booleans", () => {
  test("arithmetic", () => {
    expect(parsed("1 + 2 * 3 - 4 / 5 % 6")).toEqual(
      plain(
        new Tree("sum", [
          num("1"),
          op("ADD_OP", "+"),
          new Tree("product", [num("2"), op("MUL_OP", "*"), num("3")]),
          op("ADD_OP", "-"),
          new Tree("product", [num("4"), op("MUL_OP", "/"), num("5"), op("MUL_OP", "%"), num("6")]),
        ]),
      ),
    );
  });

  test("power", () => {
    expect(kinds(tree("2 ** 3")).has("power")).toBe(true);
  });

  test("unary minus binds looser than power", () => {
    expect(tree("-x ** 2").data).toBe("neg");
    expect(parsed("2 ** -a * b")).toEqual(
      plain(
        new Tree("product", [
          new Tree("power", [num("2"), op("POW_OP", "**"), new Tree("neg", [ref("a")])]),
          op("MUL_OP", "*"),
          ref("b"),
        ]),
      ),
    );
  });

  test("comparisons", () => {
    for (const symbol of ["==", "!=", "<", "<=", ">", ">="]) {
      expect(parsed(`a ${symbol} b`)).toEqual(plain(new Tree("compare", [ref("a"), op("COMP_OP", symbol), ref("b")])));
    }
  });

  test("and, or, not", () => {
    expect(parsed("a and b or not c")).toEqual(
      plain(
        new Tree("or_expr", [
          new Tree("and_expr", [ref("a"), op("AND", "and"), ref("b")]),
          op("OR", "or"),
          new Tree("negate", [not, ref("c")]),
        ]),
      ),
    );
  });

  test("in and not in", () => {
    expect(parsed("a in b")).toEqual(plain(new Tree("compare_in", [ref("a"), op("IN", "in"), ref("b")])));
    expect(parsed("a not in b")).toEqual(plain(new Tree("compare_notin", [ref("a"), op("NOT_IN", "not in"), ref("b")])));
    expect(parsed("a not  in b")).toEqual(plain(new Tree("compare_notin", [ref("a"), op("NOT_IN", "not  in"), ref("b")])));
    expect(parsed("x in (y)")).toEqual(plain(new Tree("compare_in", [ref("x"), op("IN", "in"), ref("y")])));
  });

  test("parens", () => {
    expect(tree("(a + b) * c").data).toBe("product");
  });

  test("prefix shapes", () => {
    expect(parsed("not x")).toEqual(plain(new Tree("negate", [not, ref("x")])));
    expect(parsed("-x")).toEqual(plain(new Tree("neg", [ref("x")])));
    expect(parsed("3 - -x")).toEqual(plain(new Tree("sum", [num("3"), op("ADD_OP", "-"), new Tree("neg", [ref("x")])])));
    expect(parsed("-inf")).toEqual(plain(new Tree("neg", [new Token("INF", "inf")])));
    expectParseErrorDetail("- not x", "unexpected 'x' at line 1, column 7");
  });

  test("a prefix inside a chain keeps the chain flat", () => {
    expect(parsed("a + -b + c")).toEqual(
      plain(new Tree("sum", [ref("a"), op("ADD_OP", "+"), new Tree("neg", [ref("b")]), op("ADD_OP", "+"), ref("c")])),
    );
    expect(parsed("not a and not b and c")).toEqual(
      plain(
        new Tree("and_expr", [
          new Tree("negate", [not, ref("a")]),
          op("AND", "and"),
          new Tree("negate", [not, ref("b")]),
          op("AND", "and"),
          ref("c"),
        ]),
      ),
    );
    expect(parsed("not a :- b")).toEqual(plain(new Tree("default_expr", [new Tree("negate", [not, ref("a")]), ref("b")])));
  });

  test("a keyword literal is an operand before an operator", () => {
    expect(parsed("inf/2")).toEqual(plain(new Tree("product", [new Token("INF", "inf"), op("MUL_OP", "/"), num("2")])));
    expect(parsed("true/x")).toEqual(plain(new Tree("product", [new Token("BOOL", "true"), op("MUL_OP", "/"), ref("x")])));
    expect(parsed("null/x")).toEqual(plain(new Tree("product", [new Token("NULL", "null"), op("MUL_OP", "/"), ref("x")])));
  });

  test("an operator word needs no boundary before it", () => {
    expect(parsed("1and 2")).toEqual(plain(new Tree("and_expr", [num("1"), op("AND", "and"), num("2")])));
    expect(parsed("1or 2")).toEqual(plain(new Tree("or_expr", [num("1"), op("OR", "or"), num("2")])));
  });

  test("not after a comparison is a name", () => {
    expect(parsed("x == not(y)")).toEqual(
      plain(new Tree("compare", [ref("x"), op("COMP_OP", "=="), call("not", [arg(ref("y"))])])),
    );
  });

  test("misplaced and chained operators are parse errors", () => {
    for (const text of [
      "x == not y", "2 ** not x", "x in not y", "not in x", "a == b == c", "a in b in c",
      "a == b in c", "a :? b :- c", "x and not", "not not", "x |", "a||b", "a :",
    ]) {
      expectParseError(text);
    }
  });
});

describe("coalesce, default and required", () => {
  test("coalesce", () => {
    expect(parsed("a | b | c")).toEqual(plain(new Tree("coalesce", [ref("a"), ref("b"), ref("c")])));
  });

  test("default non-chaining", () => {
    expectParseError("a :- b :- c");
  });

  test("required non-chaining", () => {
    expectParseError("a :? b :? c");
  });

  test("default", () => {
    expect(parsed('a :- "d"')).toEqual(plain(new Tree("default_expr", [ref("a"), new Token("STRING", '"d"')])));
  });

  test("required", () => {
    expect(parsed('a :? "msg"')).toEqual(plain(new Tree("required_expr", [ref("a"), new Token("STRING", '"msg"')])));
  });

  test("nested default is rejected", () => {
    // `a :- ${b}`: the default names `b` directly.
    expectMessage("a :- ${b}", nestedSpan("a :- b"));
  });

  test("default binds tighter than coalesce", () => {
    expect(parsed("a :- b | c")).toEqual(plain(new Tree("coalesce", [new Tree("default_expr", [ref("a"), ref("b")]), ref("c")])));
    expect(parsed("a :- b | c :- d")).toEqual(
      plain(new Tree("coalesce", [new Tree("default_expr", [ref("a"), ref("b")]), new Tree("default_expr", [ref("c"), ref("d")])])),
    );
  });

  test("coalesce binds looser than arithmetic", () => {
    expect(parsed("a | b + 1")).toEqual(
      plain(new Tree("coalesce", [ref("a"), new Tree("sum", [ref("b"), op("ADD_OP", "+"), num("1")])])),
    );
  });
});

describe("shared NAME terminal", () => {
  test("distinguishes builtin call", () => {
    expect(kinds(tree("upper(name)")).has("call_suffix")).toBe(true);
  });

  test("parses mixed call arguments", () => {
    const t = tree('join(items, ", ")');
    expect(kinds(t).has("call_suffix")).toBe(true);
    expect(kinds(t).has("arg")).toBe(true);
  });

  test("a dotted reference is one refcall", () => {
    expect(iterSubtrees(tree("a.b.c")).filter((n) => n.data === "refcall")).toHaveLength(1);
  });

  test("a hash belongs to a segment", () => {
    expect(tree("node#0.output").data).toBe("refcall");
  });

  test("a hyphen is subtraction, not part of a name", () => {
    const t = tree("a - b");
    expect(kinds(t).has("sum")).toBe(true);
    expect(iterSubtrees(t).filter((n) => n.data === "refcall")).toHaveLength(2);
  });
});

describe("literals", () => {
  test("inf is a keyword in operand position only", () => {
    expect(parsed("inf")).toEqual(plain(new Token("INF", "inf")));
    expect(parsed("x.inf")).toEqual(plain(ref("x", "inf")));
    expect(parsed("info")).toEqual(plain(ref("info")));
    expect(parsed("inf_x")).toEqual(plain(ref("inf_x")));
    for (const text of ["inf.x", "inf(x)", "inf#0"]) expectParseError(text);
  });

  test("a long integer literal parses", () => {
    expect(parsed("1".repeat(5000))).toEqual(plain(num("1".repeat(5000))));
  });

  test("digits are ASCII", () => {
    expectParseError("١");
    expectParseError("1٢");
  });

  test("keyword literals and keyword-prefixed names", () => {
    expect(parsed("none")).toEqual(plain(new Token("NULL", "none")));
    expect(parsed("null")).toEqual(plain(new Token("NULL", "null")));
    expect(parsed("true")).toEqual(plain(new Token("BOOL", "true")));
    expect(parsed("false")).toEqual(plain(new Token("BOOL", "false")));
    for (const text of ["nonex", "notx", "not#0"]) expect(parsed(text)).toEqual(plain(ref(text)));
  });

  test("numbers and strings keep their source text", () => {
    expect(parsed("01")).toEqual(plain(num("01")));
    expect(parsed("01.50")).toEqual(plain(num("01.50")));
    expect(parsed(`'a"b'`)).toEqual(plain(new Token("STRING", `'a"b'`)));
    expect(parsed('"a\nb"')).toEqual(plain(new Token("STRING", '"a\nb"')));
  });

  test("malformed literals", () => {
    for (const text of ["true.x", "true(x)", "true#0.x", "1.", ".5", "1e5", "a..b", "'abc", "1.x", "1 .5"]) {
      expectParseError(text);
    }
  });
});

describe("tokens", () => {
  test("positions are UTF-16 offsets", () => {
    const t = tree('["😀", ab.cd, 42]');
    const [str, abcd, n] = t.children as [Token, Tree, Token];
    expect([str.type, str.startPos, str.endPos]).toEqual(["STRING", 1, 5]);
    const root = abcd.children[0] as Token;
    expect([root.value, root.startPos, root.endPos]).toEqual(["ab", 7, 9]);
    const field = (abcd.children[1] as Tree).children[0] as Token;
    expect([field.value, field.startPos, field.endPos]).toEqual(["cd", 10, 12]);
    expect([n.type, n.startPos, n.endPos]).toEqual(["NUMBER", 14, 16]);
  });

  test("a token built outside the parser has no position", () => {
    const tok = new Token("NAME", "x");
    expect([tok.startPos, tok.endPos, String(tok)]).toEqual([0, 0, "x"]);
  });
});

describe("parse errors", () => {
  const cases: [string, string][] = [
    ["a)", "unexpected ')' at line 1, column 2"],
    ["x @ y", "unexpected character '@' at line 1, column 3"],
    ["1.5.2", "unexpected '.' at line 1, column 4"],
    ["a b", "unexpected 'b' at line 1, column 3"],
    ["a notx", "unexpected 'notx' at line 1, column 3"],
    ['"😀" @', "unexpected character '@' at line 1, column 6"],
    ["a\n )", "unexpected ')' at line 2, column 2"],
    ["a\r )", "unexpected ')' at line 2, column 2"],
    ["a\r\n )", "unexpected ')' at line 2, column 2"],
    ["(x", "unexpected end of expression at line 1, column 3"],
    ["x +", "unexpected end of expression at line 1, column 4"],
    ["a\n+ )", "unexpected ')' at line 2, column 3"],
    ["a\r+ )", "unexpected ')' at line 2, column 3"],
    ["a\r\n+ )", "unexpected ')' at line 2, column 3"],
    ["a == b == c", "unexpected '==' at line 1, column 8"],
    ["x + *", "unexpected '*' at line 1, column 5"],
    ["a not inx", "unexpected 'not' at line 1, column 3"],
  ];
  for (const [text, detail] of cases) {
    test(JSON.stringify(text), () => expectParseErrorDetail(text, detail));
  }
});

describe("error classes", () => {
  test("ExpressionError is a RangeError", () => {
    const error = new ExpressionError("x");
    expect(error).toBeInstanceOf(RangeError);
    expect(error.name).toBe("ExpressionError");
  });

  test("RequiredError and ConstructorTypeError extend it", () => {
    const required = new RequiredError("x");
    const constructor = new ConstructorTypeError("x");
    expect(required).toBeInstanceOf(ExpressionError);
    expect(required.name).toBe("RequiredError");
    expect(constructor).toBeInstanceOf(ExpressionError);
    expect(constructor.name).toBe("ConstructorTypeError");
  });
});

describe("iterSubtrees", () => {
  test("Lark's order: deeper first, then left to right", () => {
    const order = iterSubtrees(tree("f(g(a), h(b))")).map((n) =>
      n.data === "refcall" ? `refcall ${(n.children[0] as Token).value}` : n.data,
    );
    expect(order).toEqual([
      "refcall a",
      "refcall b",
      "arg",
      "arg",
      "call_suffix",
      "call_suffix",
      "refcall g",
      "refcall h",
      "arg",
      "arg",
      "call_suffix",
      "refcall f",
    ]);
  });
});

describe("iterSubtreesTopdown", () => {
  test("Lark's order: each node before its children, left to right", () => {
    const order = iterSubtreesTopdown(tree("f(g(a), h(b))")).map((n) =>
      n.data === "refcall" ? `refcall ${(n.children[0] as Token).value}` : n.data,
    );
    expect(order).toEqual([
      "refcall f",
      "call_suffix",
      "arg",
      "refcall g",
      "call_suffix",
      "arg",
      "refcall a",
      "arg",
      "refcall h",
      "call_suffix",
      "arg",
      "refcall b",
    ]);
  });
});

describe("long but shallow expressions", () => {
  test("parens", () => {
    expect(parsed("(".repeat(3000) + "x" + ")".repeat(3000))).toEqual(plain(ref("x")));
  });

  test("fields", () => {
    expect(tree("a" + ".b".repeat(3000)).children).toHaveLength(3001);
  });

  test("list", () => {
    expect(tree("[" + Array(3000).fill("1").join(", ") + "]").children).toHaveLength(3000);
  });

  // One flat chain; coalesce holds no operator tokens.
  const chains: [id: string, operand: string, separator: string, data: string, children: number][] = [
    ["and", "true", " and ", "and_expr", 5999],
    ["coalesce", "x", " | ", "coalesce", 3000],
    ["add", "1", " + ", "sum", 5999],
    ["sub", "1", " - ", "sum", 5999],
    ["mul", "1", " * ", "product", 5999],
    ["div", "1", " / ", "product", 5999],
    ["pow", "1", " ** ", "power", 5999],
  ];
  for (const [id, operand, separator, data, children] of chains) {
    test(id, () => {
      const t = tree(Array(3000).fill(operand).join(separator));
      expect(t.data).toBe(data);
      expect(t.children).toHaveLength(children);
    });
  }
});

describe("nesting limit", () => {
  afterEach(() => install(BUILTIN));

  const tooDeep = (limit: number) =>
    `expression nests deeper than ${limit} levels, the limit for a yian expression (each call, list, ` +
    "unary operator, comparison, and chain of operators is a level); split it into smaller bindings or steps";

  /** `((1 + 1) * 2) + 1 ...`: each switch of precedence level is a level. */
  function alternating(n: number): string {
    let text = "1 + 1";
    for (let i = 0; i < n - 1; i++) text = i % 2 === 0 ? `(${text}) * 2` : `(${text}) + 1`;
    return text;
  }

  /** `((1 + 1) + 1) + 1 ...`: author parentheses make each `+` its own chain. */
  function grouped(n: number): string {
    let text = "1 + 1";
    for (let i = 0; i < n - 1; i++) text = `(${text}) + 1`;
    return text;
  }

  // kind -> a builder of an expression exactly `n` levels deep.
  const nested = {
    not: (n) => "not ".repeat(n) + "true",
    neg: (n) => "-(".repeat(n) + "1" + ")".repeat(n),
    list: (n) => "[".repeat(n) + "1" + "]".repeat(n),
    call: (n) => "upper(".repeat(n) + "x" + ")".repeat(n),
    and: (n) => "true and (".repeat(n - 1) + "true and true" + ")".repeat(n - 1),
    coalesce: (n) => "x | (".repeat(n - 1) + "x | 1" + ")".repeat(n - 1),
    default: (n) => "x :- (".repeat(n - 1) + "x :- 1" + ")".repeat(n - 1),
    compare: (n) => "(".repeat(n - 1) + "1 == 1" + ") == true".repeat(n - 1),
    index: (n) => "x" + "[0]".repeat(n),
    conditional: (n) => "1 if c else (".repeat(n - 1) + "1 if c else 2" + ")".repeat(n - 1),
    alternating,
    grouped,
  } satisfies Record<string, (n: number) => string>;

  for (const [kind, build] of Object.entries(nested)) {
    test(`${kind} at the nesting limit parses`, () => {
      expect(() => parseExpr(build(100))).not.toThrow();
    });

    for (const depth of [101, 3000]) {
      test(`${kind} ${depth} deep is an expression error`, () => {
        expectMessage(build(depth), tooDeep(100));
      });
    }
  }

  test("the limit is current().maxValueNesting", () => {
    install({ ...BUILTIN, maxValueNesting: 5 });
    expect(() => parseExpr(nested.not(5))).not.toThrow();
    expectMessage(nested.not(6), tooDeep(5));
  });

  test("a malformed too-deep expression is the parse error", () => {
    expectParseError("not ".repeat(3000) + "true +");
  });
});

describe("float overflow", () => {
  const overflow = (digits: number) =>
    `float literal with ${digits} digits before the decimal point is out of range; write inf for infinity`;

  test("a float literal past the double range is an expression error", () => {
    expectMessage("1".repeat(400) + ".0", overflow(400));
  });

  test("the rightmost overflowing literal is reported", () => {
    expectMessage("1".repeat(400) + ".0 + " + "1".repeat(500) + ".0", overflow(500));
  });

  test("leading zeros don't count", () => {
    expect(parsed("0".repeat(400) + "1.5")).toEqual(plain(num("0".repeat(400) + "1.5")));
  });

  test("the edge of the double range", () => {
    expectMessage("9".repeat(309) + ".0", overflow(309));
    expect(() => parseExpr("9".repeat(308) + ".0")).not.toThrow();
  });
});

describe("keywords and bare names", () => {
  test("EXPRESSION_KEYWORDS", () => {
    expect(EXPRESSION_KEYWORDS).toEqual(new Set(["and", "or", "not", "in", "true", "false", "null", "none", "inf"]));
  });

  test("isBareName", () => {
    expect(isBareName(parseExpr("x"))).toBe(true);
    for (const text of ["x.y", "f(x)", "1"]) expect(isBareName(parseExpr(text))).toBe(false);
    expect(isBareName(new Tree("refcall", [new Token("NAME", "and")]))).toBe(false);
  });
});

describe("check order", () => {
  const isinstanceArity = "isinstance takes exactly two arguments (a value and a type name), got 1";
  const typeName = "isinstance's second argument must be a bare type name, e.g. isinstance(x, Buy)";
  const cases: [string, string][] = [
    ["isinstance(a.b(x), T)", DOTTED_CALLEE],
    ["a.b(isinstance(x))", DOTTED_CALLEE],
    ["isinstance(x, in)", typeName],
    ["isinstance(x) and isinstance(y, 1)", isinstanceArity],
    ["isinstance(y, 1) and isinstance(x)", typeName],
    ["isinstance(b) and f(isinstance(a, 1))", typeName],
    ["f(a=1, a=2) and g(b=1,b=2)", "f(...) repeats the keyword 'a'"],
    ["f(a=1,a=2) + g(h(b=1,b=2))", "h(...) repeats the keyword 'b'"],
    ["f(a=1, b=1, b=2, a=2)", "f(...) repeats the keyword 'b'"],
    ["isinstance(y) + a.b(x)", DOTTED_CALLEE],
  ];
  for (const [text, message] of cases) {
    test(text, () => expectMessage(text, message));
  }
});

describe("span helpers", () => {
  test("findSpanEnd", () => {
    // Quote aware: the `}` in the string literal doesn't close the span.
    expect(findSpanEnd('x == "}"}', 0)).toBe(8);
    expect(findSpanEnd("a{b}", 0)).toBeNull();
    expect(findSpanEnd("a{b}}", 0)).toBe(4);
    expect(findSpanEnd('x {"{"} }tail', 0)).toBe(8);
  });

  test("wholeSpanInterior", () => {
    expect(wholeSpanInterior(" ${a} ")).toBe("a");
    expect(wholeSpanInterior('${x == "}"}')).toBe('x == "}"');
    expect(wholeSpanInterior("${a} > ${b}")).toBeNull();
    expect(wholeSpanInterior("a")).toBeNull();
    expect(wholeSpanInterior("${a")).toBeNull();
  });
});

describe("subscripts", () => {
  const omitted = () => new Tree("omitted", []);
  const sub = (base: ExprNode, lower: ExprNode, upper: ExprNode) =>
    new Tree("subscript", [base, new Tree("slice", [lower, upper])]);
  /** The rule name of `node`, or its token type. */
  const kindOf = (node: ExprNode) => (node instanceof Tree ? node.data : node.type);
  /** The `slice` of the subscript `text` parses to. */
  const sliceOf = (text: string) => {
    const result = tree(text);
    expect(result.data).toBe("subscript");
    return result.children[1] as Tree;
  };

  test.each([
    ["xs[1:]", sub(ref("xs"), num("1"), omitted())],
    ["xs[:-1]", sub(ref("xs"), omitted(), new Tree("neg", [num("1")]))],
    ["xs[:]", sub(ref("xs"), omitted(), omitted())],
    ["xs[a :- b]", sub(ref("xs"), ref("a"), new Tree("neg", [ref("b")]))],
  ])("%s parses as a slice", (text, expected) => expect(parsed(text)).toEqual(plain(expected)));

  test.each([
    ["xs[0]", new Tree("subscript", [ref("xs"), new Tree("index", [num("0")])])],
    ["xs[-1]", new Tree("subscript", [ref("xs"), new Tree("index", [new Tree("neg", [num("1")])])])],
    [
      "xs[(i :- 0)]",
      new Tree("subscript", [ref("xs"), new Tree("index", [new Tree("default_expr", [ref("i"), num("0")])])]),
    ],
  ])("%s parses as an index", (text, expected) => expect(parsed(text)).toEqual(plain(expected)));

  test("an index of an index, and of a slice", () => {
    expect(tree("xs[0][1]").children.map(kindOf)).toEqual(["subscript", "index"]);
    expect(tree("xs[1:][0]").children.map(kindOf)).toEqual(["subscript", "index"]);
  });

  test("an index binds tighter than unary minus and **", () => {
    expect(tree("-xs[0]").children.map(kindOf)).toEqual(["subscript"]);
    expect(tree("xs[0] ** 2").children.map(kindOf)).toEqual(["subscript", "POW_OP", "NUMBER"]);
  });

  test("a subscript of a subscript", () => {
    const result = tree("xs[1:][:1]");
    expect(result.data).toBe("subscript");
    expect(kindOf(result.children[0]!)).toBe("subscript");
  });

  test("a subscript binds tighter than unary minus and **", () => {
    const negated = tree("-xs[1:]");
    expect(negated.data).toBe("neg");
    expect(negated.children.map(kindOf)).toEqual(["subscript"]);
    const power = tree("xs[1:] ** 2");
    expect(power.data).toBe("power");
    expect(power.children.map(kindOf)).toEqual(["subscript", "POW_OP", "NUMBER"]);
  });

  test.each([
    ["xs[n - 1:-1]", "sum"],
    ["xs[a * 2:-b]", "product"],
  ])("in %s a pending operator doesn't hide the slice's colon", (text, lower) => {
    expect(sliceOf(text).children.map(kindOf)).toEqual([lower, "neg"]);
  });

  test("inside parentheses or a list, :- is still a default", () => {
    expect(sliceOf("xs[(n :- 0):]").children.map(kindOf)).toEqual(["default_expr", "omitted"]);
    const list = tree("[n :- 0]");
    expect(list.data).toBe("list_lit");
    expect(list.children.map(kindOf)).toEqual(["default_expr"]);
  });

  test("a long chain of prefixes parses in linear time", () => {
    // Each `-` stays a pending frame, so a parse that scanned the stack per operator would be
    // quadratic. Compare 8x the terms against the fastest of three runs each, which keeps the
    // check about growth, not about the machine's speed.
    const fastest = (terms: number) => {
      const text = Array(terms).fill("-a").join(" ** ");
      let best = Infinity;
      for (let i = 0; i < 3; i++) {
        const start = performance.now();
        expect(() => parseExpr(text)).toThrow("expression nests deeper than");
        best = Math.min(best, performance.now() - start);
      }
      return best;
    };
    const small = fastest(4096);
    // Linear growth gives about 8x; quadratic gives about 64x.
    expect(fastest(32768) / Math.max(small, 1)).toBeLessThan(24);
  });

  test("a call's repeated-keyword check is linear in its keywords", () => {
    // A check that scanned the keywords once per keyword would be quadratic. Compare 8x the
    // keywords against the fastest of three runs each, as above.
    const fastest = (keywords: number) => {
      const text = `Point(${Array.from({ length: keywords }, (_, i) => `f${i}=1`).join(", ")})`;
      let best = Infinity;
      for (let i = 0; i < 3; i++) {
        const start = performance.now();
        parseExpr(text);
        best = Math.min(best, performance.now() - start);
      }
      return best;
    };
    const small = fastest(2000);
    // Linear growth gives about 8x; quadratic gives about 64x.
    expect(fastest(16000) / Math.max(small, 1)).toBeLessThan(24);
  });

  test.each([
    ["f(x)[1:]", "refcall"],
    ["(xs + ys)[1:]", "sum"],
    ["[1, 2, 3][1:]", "list_lit"],
  ])("%s subscripts any operand", (text, base) => {
    const result = tree(text);
    expect(result.data).toBe("subscript");
    expect(kindOf(result.children[0]!)).toBe(base);
    if (base === "refcall") expect((result.children[0] as Tree).children.map(kindOf)).toContain("call_suffix");
  });
});

describe("comprehensions", () => {
  /** The rule name of `node`, or its token type. */
  const kindOf = (node: ExprNode) => (node instanceof Tree ? node.data : node.type);
  /** Each clause of the comprehension `text` parses to, as its kind and, for a `for`, target and source. */
  const clausesOf = (text: string) => {
    const result = tree(text);
    expect(result.data).toBe("list_comp");
    return (result.children.slice(1) as Tree[]).map((clause) =>
      clause.data === "comp_for" ? [clause.data, String(clause.children[0]), String(clause.children[2])] : [clause.data],
    );
  };

  test("a map is a list_comp of its elem and one for", () => {
    const result = tree("[x * 2 for x in xs]");
    expect(result.data).toBe("list_comp");
    expect(kindOf(result.children[0]!)).toBe("product");
    expect(clausesOf("[x * 2 for x in xs]")).toEqual([["comp_for", "x", "xs"]]);
    const source = (result.children[1] as Tree).children[2] as Token;
    expect(source.type).toBe("SOURCE");
  });

  test("clauses keep source order", () => {
    expect(clausesOf('[m for t in ts for m in t if m.role == "user"]')).toEqual([
      ["comp_for", "t", "ts"],
      ["comp_for", "m", "t"],
      ["comp_if"],
    ]);
    expect(clausesOf("[x for x in xs if x > 0 if x < 3]")).toEqual([["comp_for", "x", "xs"], ["comp_if"], ["comp_if"]]);
  });

  test("an elem may be a comprehension or an escape", () => {
    expect(kindOf(tree("[[c for c in r] for r in rows]").children[0]!)).toBe("list_comp");
    expect(kindOf(tree("[a :- b for x in xs]").children[0]!)).toBe("default_expr");
  });

  test("SOURCE keeps the iterable's text, parentheses included", () => {
    expect(clausesOf("[x for x in (a :- [])]")).toEqual([["comp_for", "x", "(a :- [])"]]);
  });

  test("for and if stay names in operand position", () => {
    const list = tree("[for, if]");
    expect(list.data).toBe("list_lit");
    expect(list.children.map(kindOf)).toEqual(["refcall", "refcall"]);
    expect(parsed("for.x")).toEqual(plain(ref("for", "x")));
  });

  test("5,000 if clauses parse, with no recursion and no clause cap", () => {
    const result = tree(`[x for x in xs${" if true".repeat(5000)}]`);
    expect(result.children.length).toBe(5002);
  });
});

describe("conditionals", () => {
  const cond = (then: ExprNode, test: ExprNode, otherwise: ExprNode) => new Tree("if_expr", [then, test, otherwise]);
  /** The rule name of `node`, or its token type. */
  const kindOf = (node: ExprNode) => (node instanceof Tree ? node.data : node.type);

  test.each([
    ["a if c else b", cond(ref("a"), ref("c"), ref("b"))],
    // The conditional is the lowest level: an escape belongs to a branch.
    ["x :- y if c else z", cond(new Tree("default_expr", [ref("x"), ref("y")]), ref("c"), ref("z"))],
    ["x | y if c else z", cond(new Tree("coalesce", [ref("x"), ref("y")]), ref("c"), ref("z"))],
    ["a if c else x :- y", cond(ref("a"), ref("c"), new Tree("default_expr", [ref("x"), ref("y")]))],
    ["a if c :- d else b", cond(ref("a"), new Tree("default_expr", [ref("c"), ref("d")]), ref("b"))],
    ["a if c else b if d else e", cond(ref("a"), ref("c"), cond(ref("b"), ref("d"), ref("e")))],
    ["(a if c else b) if d else e", cond(cond(ref("a"), ref("c"), ref("b")), ref("d"), ref("e"))],
    ["a if not c else b", cond(ref("a"), new Tree("negate", [not, ref("c")]), ref("b"))],
  ])("%s", (text, expected) => expect(parsed(text)).toEqual(plain(expected)));

  test("operators bind tighter than a conditional, on both sides", () => {
    expect(tree("a + 1 if c or d else b * 2").children.map(kindOf)).toEqual(["sum", "or_expr", "product"]);
  });

  test("in a comprehension, a conditional is the elem and `if` after a `for` a filter", () => {
    const result = tree("[a if c else b for x in xs if x]");
    expect(result.data).toBe("list_comp");
    expect(result.children.map(kindOf)).toEqual(["if_expr", "comp_for", "comp_if"]);
  });

  test("a conditional is a list item, a call argument, a slice bound and an index", () => {
    expect(tree("[a if c else b, d]").children.map(kindOf)).toEqual(["if_expr", "refcall"]);
    const args = (tree("f(a if c else b)").children[1] as Tree).children as Tree[];
    expect(args.map((x) => kindOf(x.children[0]!))).toEqual(["if_expr"]);
    expect((tree("xs[a if c else b:]").children[1] as Tree).children.map(kindOf)).toEqual(["if_expr", "omitted"]);
    expect((tree("xs[a if c else b]").children[1] as Tree).children.map(kindOf)).toEqual(["if_expr"]);
  });

  test.each<[string, string]>([
    ["a if c", "a conditional needs `else`: write `a if c else b` at line 1, column 7"],
    ["[a if c for x in xs]", "a conditional needs `else`: write `a if c else b` at line 1, column 9"],
    ["f(a if c)", "a conditional needs `else`: write `a if c else b` at line 1, column 9"],
    [
      "a if b if c else d else e",
      "a conditional's test can't hold another conditional; write `a if (b if c else d) else e` at line 1, column 8",
    ],
    ["a else b", "`else` comes after `a if c`, as in `a if c else b` at line 1, column 3"],
    ["a if c else b else d", "`else` comes after `a if c`, as in `a if c else b` at line 1, column 15"],
    [
      "[x for x in xs if a else b]",
      "a comprehension's `if` is a filter; put a conditional there in parentheses: `(a if c else b)` at line 1, column 21",
    ],
  ])("%s fails with the fix", (text, detail) => {
    expectParseErrorDetail(text, detail);
  });

  test("if and else are names in operand position and after a dot", () => {
    expect(parsed("if if else else else")).toEqual(plain(cond(ref("if"), ref("else"), ref("else"))));
    expect(parsed("x.if")).toEqual(plain(ref("x", "if")));
    expect(parsed("x.else")).toEqual(plain(ref("x", "else")));
    expect(tree("[else, if]").children.map(kindOf)).toEqual(["refcall", "refcall"]);
  });

  test("else can't be a comprehension variable", () => {
    expectParseErrorDetail("[x for else in xs]", "a comprehension's `for` takes a name, then `in` at line 1, column 8");
  });
});

describe("comprehension scoping", () => {
  /** The `comp_var` nodes in `node`, each as its plain form. */
  const compVars = (node: Tree) => iterSubtrees(node).filter((n) => n.data === "comp_var").map(plain);
  const compVar = (...parts: string[]) => plain(new Tree("comp_var", ref(...parts).children));
  /** The `refcall` nodes in `node`, each as its plain form. */
  const refcalls = (node: Tree) => iterSubtrees(node).filter((n) => n.data === "refcall").map(plain);

  test("the elem's variable is a comp_var and the iterable a reference", () => {
    const result = tree("[x * 2 for x in xs]");
    expect(compVars(result)).toEqual([compVar("x")]);
    expect(plain((result.children[1] as Tree).children[1]!)).toEqual(plain(ref("xs")));
  });

  test("a later iterable and condition see earlier variables", () => {
    const result = tree('[m for t in ts for m in t if m.role == "user"]');
    expect(plain((result.children[2] as Tree).children[1]!)).toEqual(compVar("t"));
    expect(compVars(result)).toContainEqual(compVar("m", "role"));
    expect(refcalls(result)).toEqual([plain(ref("ts"))]);
  });

  test("the first iterable runs in the enclosing scope", () => {
    const result = tree("[x for x in x.items]");
    expect(plain((result.children[1] as Tree).children[1]!)).toEqual(plain(ref("x", "items")));
    expect(compVars(result)).toEqual([compVar("x")]);
  });

  test("a nested comprehension sees the enclosing variables", () => {
    const result = tree("[[c + r0 for c in r] for r in rows]");
    const inner = result.children[0] as Tree;
    expect(compVars(inner).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)))).toEqual([
      compVar("c"),
      compVar("r"),
    ]);
    expect(refcalls(result)).toEqual([plain(ref("r0")), plain(ref("rows"))]);
  });

  test("isinstance's type name stays a name", () => {
    expect(compVars(tree("[isinstance(T, T) for T in ts]"))).toEqual([compVar("T")]);
  });

  test("a call is never a variable read", () => {
    const result = tree('[len(len) for len in ["ab"]]');
    expect(plain(result.children[0]!)).toEqual(plain(call("len", [arg(new Tree("comp_var", [name("len")]))])));
  });

  test("an inner comprehension in the first iterable may reuse the name", () => {
    expect(tree("[x for x in [x for x in a]]").data).toBe("list_comp");
  });
});

test("two comprehensions binding m give their reads different keys", () => {
  const t = tree("[[m.a for m in xs], [m.b for m in ys]]");
  const keys = iterSubtrees(t).filter((n) => n.data === "comp_var").map(compVarKey);
  expect(new Set(keys).size).toBe(2);
  expect(keys.every((key) => key !== null && key.startsWith("m@"))).toBe(true);
  expect(compVarKey(new Tree("comp_var", [new Token("NAME", "m")]))).toBeNull();
});

test("reads of one variable in nested scopes share its key", () => {
  const t = tree("[[n for n in m.xs if m.ok] for m in ys]");
  const keys = iterSubtrees(t)
    .filter((n) => n.data === "comp_var")
    .map(compVarKey);
  expect(new Set(keys)).toEqual(new Set(["m@31", "n@8"]));
});

test("a copied comp_var keeps its key, as masked and falsified copies do", () => {
  const t = tree("[m.a for m in xs]");
  const node = iterSubtrees(t).find((n) => n.data === "comp_var")!;
  // masked/falsified rebuild each Tree but keep its Tokens.
  expect(compVarKey(new Tree(node.data, [...node.children]))).toBe(compVarKey(node));
});
