/**
 * Operator chains are flat nodes: a port of the "Parse shapes" part of
 * `tests/engine/test_expr_chains.py`. The evaluation and lineage halves land with the
 * evaluator; `test_the_grammar_has_no_lalr_conflict` has no counterpart, since the TS parser
 * has no grammar table.
 */

import { expect, test } from "bun:test";

import { type ExprNode, parseExpr, Token, Tree } from "../../src/expr/grammar.ts";

/** A position-blind form of `node`, as Lark's `Tree`/`Token` equality ignores positions. */
type Plain = { type: string; value: string } | { data: string; children: Plain[] };

function plain(node: ExprNode): Plain {
  if (node instanceof Token) return { type: node.type, value: node.value };
  return { data: node.data, children: node.children.map(plain) };
}

const ref = (name: string) => new Tree("refcall", [new Token("NAME", name)]);
const num = (text: string) => new Token("NUMBER", text);
const add = (symbol: string) => new Token("ADD_OP", symbol);
const mul = (symbol: string) => new Token("MUL_OP", symbol);
const POW = new Token("POW_OP", "**");

const cases: [string, ExprNode][] = [
  ["a + b - c", new Tree("sum", [ref("a"), add("+"), ref("b"), add("-"), ref("c")])],
  [
    "a * b / c % d",
    new Tree("product", [ref("a"), mul("*"), ref("b"), mul("/"), ref("c"), mul("%"), ref("d")]),
  ],
  ["a ** b ** c", new Tree("power", [ref("a"), POW, ref("b"), POW, ref("c")])],
  ["a + b * c", new Tree("sum", [ref("a"), add("+"), new Tree("product", [ref("b"), mul("*"), ref("c")])])],
  ["-a ** 2", new Tree("neg", [new Tree("power", [ref("a"), POW, num("2")])])],
  [
    "2 ** -a ** b",
    new Tree("power", [num("2"), POW, new Tree("neg", [new Tree("power", [ref("a"), POW, ref("b")])])]),
  ],
  ["a - -b", new Tree("sum", [ref("a"), add("-"), new Tree("neg", [ref("b")])])],
  ["a--b", new Tree("sum", [ref("a"), add("-"), new Tree("neg", [ref("b")])])],
  ["1 -1", new Tree("sum", [num("1"), add("-"), num("1")])],
  ["a*-b", new Tree("product", [ref("a"), mul("*"), new Tree("neg", [ref("b")])])],
  ["(a + b) + c", new Tree("sum", [new Tree("sum", [ref("a"), add("+"), ref("b")]), add("+"), ref("c")])],
  ["a + (b + c)", new Tree("sum", [ref("a"), add("+"), new Tree("sum", [ref("b"), add("+"), ref("c")])])],
  ["a/b", ref("a/b")],
];

for (const [text, expected] of cases) {
  test(`chain shape ${JSON.stringify(text)}`, () => {
    expect(plain(parseExpr(text))).toEqual(plain(expected));
  });
}
