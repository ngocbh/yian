/**
 * Parse tests for the unified `${...}` expression grammar (`expr/grammar.ts`), a port of
 * `tests/engine/test_expr_grammar.py`: references, operators and their precedence, builtin
 * calls, keyword arguments, list literals, groups, coalesce/default/required, literals, token
 * positions, parse-error messages, the error classes and `iterSubtrees` order. Later: the
 * nesting-limit tests come with the post-parse checks (dotted callee, reserved roots), and the
 * nested-span and `scanTemplate` tests with the `${...}` helpers and `template.ts`.
 */

import { describe, expect, test } from "bun:test";

import { ConstructorTypeError, ExpressionError, RequiredError } from "../../src/expr/expressions.ts";
import { type ExprNode, iterSubtrees, parseExpr, Token, Tree } from "../../src/expr/grammar.ts";

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

  test("keyword prefixes stay legal in generated roots", () => {
    for (const root of ["and/child", "or#0", "not/child", "not#0", "in#0"]) {
      expect(parsed(`${root}.output`)).toEqual(plain(ref(root, "output")));
    }
  });

  test("keywords are legal dotted fields", () => {
    for (const field of ["and", "or", "not", "in", "true", "inf"]) {
      expect(parsed(`record.output.${field}`)).toEqual(plain(ref("record", "output", field)));
    }
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
      "- not x", "x == not y", "2 ** not x", "x in not y", "not in x", "a == b == c", "a in b in c",
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

  test("default and required don't chain", () => {
    expectParseError("a :- b :- c");
    expectParseError("a :? b :? c");
  });

  test("default", () => {
    expect(parsed('a :- "d"')).toEqual(plain(new Tree("default_expr", [ref("a"), new Token("STRING", '"d"')])));
  });

  test("required", () => {
    expect(parsed('a :? "msg"')).toEqual(plain(new Tree("required_expr", [ref("a"), new Token("STRING", '"msg"')])));
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
  test("a call is told from a reference", () => {
    expect(kinds(tree("upper(name)")).has("call_suffix")).toBe(true);
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

  const chains: [string, string, string][] = [
    ["and", "true", " and "],
    ["coalesce", "x", " | "],
    ["add", "1", " + "],
    ["sub", "1", " - "],
    ["mul", "1", " * "],
    ["div", "1", " / "],
    ["pow", "1", " ** "],
  ];
  for (const [id, operand, separator] of chains) {
    test(id, () => {
      const t = tree(Array(3000).fill(operand).join(separator));
      // One flat chain; coalesce holds no operator tokens.
      expect(t.children).toHaveLength(id === "coalesce" ? 3000 : 5999);
    });
  }
});
