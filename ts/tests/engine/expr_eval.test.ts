/**
 * The one unified expression evaluator (`evalExpr`), a port of `tests/engine/test_expr_eval.py`.
 *
 * These drive `evalExpr(tree, resolve, {item, mode})`, the single evaluator that walks a tree
 * from `parseExpr` against a `resolve` callable. They check computed values, the three
 * missing-reference behaviors by mode, and the semantics a CASE's `else:` routing depends on
 * (a miss is `null`, arithmetic over it raises, an ordered comparison with it is `false`), then
 * the TS-era rules: the number model (an int is a `bigint` at any size, `/` always gives a
 * correctly rounded float), the operator table, and the messages.
 *
 * The shipped builtins (`upper`, `lower`, `join`, `render_as_json`) run as themselves, with TS-era
 * rows for the text form `join` writes and the layout `render_as_json` writes; calls to any other
 * name run through stand-ins registered for one test each. `test_binding_default_ref` and
 * `test_nested_default_span_rejected` run through `evalBinding` and `scanTemplate`. Python's `%`
 * formatting, repetition, overflow and digit-limit tests become the TS-era tables below;
 * `_Boom` (a value whose negation raises) has no counterpart, since a value is plain data.
 *
 * TS-only: a long list comprehension evaluates without recursion.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import {
  declaredAt,
  ExpressionError,
  evalExpr,
  exprRefs,
  localDeclared,
  MISSING_ROOT,
  RequiredError,
  ResolveMode,
  type RootResolver,
  rewriteExprRefs,
} from "../../src/expr/expressions.ts";
import { type ExprNode, parseExpr, Token, Tree } from "../../src/expr/grammar.ts";
import { evalBinding, scanTemplate } from "../../src/expr/template.ts";
import { FileRef, type Value, type ValueRecord } from "../../src/typesys/index.ts";
import { fieldOf, Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";

/** Parse `text` then evaluate it, defaulting to an empty resolver (every reference misses). */
function evaluate(
  text: string,
  resolve: RootResolver = MISSING_ROOT,
  { item = null, mode = ResolveMode.BINDING_NONE }: { item?: Value; mode?: ResolveMode } = {},
): Value {
  return evalExpr(parseExpr(text), resolve, { item, mode });
}

/**
 * Return a `resolve` backed by a record, with the field-only dotted walk the production
 * resolvers do: the head, then `fieldOf` per step, never a property read. A miss is `null`.
 */
function fromRecord(mapping: ValueRecord): RootResolver {
  return (names) => {
    const [present, value] = fieldOf(mapping, names[0]!);
    return { used: 1, present, value, why: null };
  };
}

/** Return the error `evaluate(text, resolve, options)` throws, failing if it returns. */
function errorOf(text: string, resolve?: RootResolver, mode?: ResolveMode): Error {
  try {
    evaluate(text, resolve, { mode });
  } catch (error) {
    return error as Error;
  }
  throw new Error(`${text} evaluated without an error`);
}

/** Expect `text` to fail with an `ExpressionError` whose message is exactly `message`. */
function expectMessage(text: string, message: string, resolve?: RootResolver): void {
  const error = errorOf(text, resolve);
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message).toBe(message);
}

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in builtin `name` taking `params` and running `call`, for this test only. */
function standIn(name: string, params: TemplateParam[], call: (...args: Value[]) => unknown): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

// An int's digits at the shown-value cap: 199 characters and an ellipsis.
const BIG_SHOWN = "1" + "0".repeat(198) + "…";

// --------------------------------------------------------------------------------------- //
// Operators on resolved values
// --------------------------------------------------------------------------------------- //

test("int addition", () => {
  expect(evaluate("1 + 2")).toBe(3n);
});

test("string concatenation", () => {
  expect(evaluate('"a" + "b"')).toBe("ab");
});

test("list concatenation", () => {
  expect(evaluate("x + y", fromRecord({ x: [1n], y: [2n] }))).toEqual([1n, 2n]);
});

test("power", () => {
  expect(evaluate("2 ** 3")).toBe(8n);
});

test("subtraction, product, division and remainder", () => {
  expect(evaluate("10 - 3")).toBe(7n);
  expect(evaluate("4 * 5")).toBe(20n);
  expect(evaluate("8 / 2")).toBe(4);
  expect(evaluate("7 % 3")).toBe(1n);
});

test("unary minus", () => {
  expect(evaluate("-x", fromRecord({ x: 5n }))).toBe(-5n);
});

// --------------------------------------------------------------------------------------- //
// Operator errors name the operator and both values
// --------------------------------------------------------------------------------------- //

test("a type error with a huge int shows its digits cut to the cap", () => {
  expectMessage(
    'n + "a"',
    `type error evaluating ${BIG_SHOWN} + "a": unsupported operand kinds for \`+\`: \`int\` and \`str\``,
    fromRecord({ n: 10n ** 5000n }),
  );
});

test("division by zero with a huge int", () => {
  expectMessage("n / 0", `division by zero: ${BIG_SHOWN} / 0`, fromRecord({ n: 10n ** 5000n }));
});

test("a comparison type error with a huge int", () => {
  expectMessage(
    'n < "a"',
    `type error evaluating ${BIG_SHOWN} < "a": unsupported operand kinds for \`<\`: \`int\` and \`str\``,
    fromRecord({ n: 10n ** 5000n }),
  );
});

test("unary minus on a list holding a huge int", () => {
  expectMessage(
    "-xs",
    `unary minus on non-number [1${"0".repeat(197)}…: unsupported operand kind for \`-\`: \`list\``,
    fromRecord({ xs: [10n ** 5000n] }),
  );
});

test("the type error message", () => {
  expectMessage('1 + "a"', 'type error evaluating 1 + "a": unsupported operand kinds for `+`: `int` and `str`');
});

// --------------------------------------------------------------------------------------- //
// Dotted access reads fields only, never a property (safety-critical)
// --------------------------------------------------------------------------------------- //

test("dotted access reads a record field", () => {
  expect(evaluate("x.k", fromRecord({ x: { k: 42n } }))).toBe(42n);
});

test.each(["__class__", "constructor", "toString", "__proto__", "hasOwnProperty"])(
  "dotted access never reads the property %s",
  (name) => {
    expect(evaluate(`obj.${name}`, fromRecord({ obj: { a: 1n } }))).toBeNull();
    expect(evaluate(`item.${name}`, MISSING_ROOT, { item: { a: 1n } })).toBeNull();
    expect(evaluate(`item.k.${name}`, MISSING_ROOT, { item: { k: "text" } })).toBeNull();
  },
);

// --------------------------------------------------------------------------------------- //
// Missing-reference behavior by resolution mode
// --------------------------------------------------------------------------------------- //

test("binding-none: a miss is null", () => {
  expect(evaluate("a", undefined, { mode: ResolveMode.BINDING_NONE })).toBeNull();
});

test("condition-falsy: an ordered comparison with a miss is false", () => {
  expect(evaluate("missing > 5", undefined, { mode: ResolveMode.CONDITION_FALSY })).toBe(false);
});

test("condition-falsy: arithmetic over a miss is loud", () => {
  expect(errorOf("missing + 1 > 5", undefined, ResolveMode.CONDITION_FALSY)).toBeInstanceOf(ExpressionError);
});

test("strict-raise: a miss raises, naming the path", () => {
  const error = errorOf("a.b", undefined, ResolveMode.STRICT_RAISE);
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message).toBe("unresolved reference ${a.b} in expression");
  expect(errorOf("item.k", undefined, ResolveMode.STRICT_RAISE).message).toBe(
    "unresolved reference ${item.k} in expression",
  );
});

test("strict-raise: a slice of null names the slice, not a miss an operator handled", () => {
  const strict = (text: string) =>
    errorOf(text, fromRecord({ xs: [null] }), ResolveMode.STRICT_RAISE).message;
  expect(strict("(a :- null)[0:1]")).toBe("unresolved reference ${(...)[...]} in expression");
  expect(strict("(a | null)[0:1]")).toBe("unresolved reference ${(...)[...]} in expression");
  expect(strict("(null)[0:1]")).toBe("unresolved reference ${null[...]} in expression");
  expect(strict("[x[1:] for x in xs]")).toBe("unresolved reference ${x[...]} in expression");
});

test("ResolveMode holds Python's values and is frozen", () => {
  expect({ ...ResolveMode }).toEqual({
    STRICT_RAISE: "strict-raise",
    BINDING_NONE: "binding-none",
    CONDITION_FALSY: "condition-falsy",
  });
  expect(Object.isFrozen(ResolveMode)).toBe(true);
});

// --------------------------------------------------------------------------------------- //
// Coalesce / default / required
// --------------------------------------------------------------------------------------- //

test("coalesce takes the first non-null", () => {
  expect(evaluate("a | b | c", fromRecord({ b: "second", c: "third" }))).toBe("second");
});

test("a present falsy value wins the coalesce", () => {
  expect(evaluate("a | b", fromRecord({ a: 0n }))).toBe(0n);
});

test("coalesce short-circuits", () => {
  expect(evaluate("a | 1 / 0", fromRecord({ a: "present" }))).toBe("present");
});

test("default when missing", () => {
  expect(evaluate('a :- "d"')).toBe("d");
});

test("default short-circuits its fallback", () => {
  expect(evaluate("a :- 1 / 0", fromRecord({ a: "present" }))).toBe("present");
});

test("default to a reference", () => {
  expect(evaluate("a :- b", fromRecord({ b: "fallback" }))).toBe("fallback");
});

test("a binding's default reference", () => {
  expect(evalBinding("${x :- y}", fromRecord({ y: 9n }))).toBe(9n);
  // Both missing: null in the binding mode.
  expect(evalBinding("${x :- y}", fromRecord({}))).toBeNull();
});

test("a nested default span is rejected", () => {
  // A `${...}` can't appear inside an expression, at any depth; group with parentheses.
  expect(() => scanTemplate("${x:-${y:-${z}}}")).toThrow(ExpressionError);
  expect(() => scanTemplate("${x:-${y:-${z}}}")).toThrow("write `${x:-(y:-z)}`");
});

test("required raises when missing, with its message", () => {
  const error = errorOf('a :? "m"');
  expect(error).toBeInstanceOf(RequiredError);
  expect(error.message).toBe("m");
  // A message that isn't text shows in its text form.
  expect(errorOf('a :? [1, "x"]').message).toBe('[1,"x"]');
});

test("required short-circuits its message", () => {
  expect(evaluate("a :? 1 / 0", fromRecord({ a: "here" }))).toBe("here");
});

// --------------------------------------------------------------------------------------- //
// Builtin dispatch through TEMPLATE_FNS
// --------------------------------------------------------------------------------------- //

test("upper", () => {
  expect(evaluate("upper(name)", fromRecord({ name: "hi" }))).toBe("HI");
});

test("join", () => {
  expect(evaluate('join(items, ", ")', fromRecord({ items: ["a", "b"] }))).toBe("a, b");
});

test("a keyword argument", () => {
  expect(evaluate("upper(s=name)", fromRecord({ name: "hi" }))).toBe("HI");
});

test.each<[string, ValueRecord, string]>([
  ["upper(name)", {}, 'upper() got null; give a default, e.g. `upper(x :- "")`'],
  ["lower(name)", {}, 'lower() got null; give a default, e.g. `lower(x :- "")`'],
  ["upper(n)", { n: 3n }, "upper() takes str, got int"],
  ["lower(xs)", { xs: ["A"] }, "lower() takes str, got list"],
  ["render_as_json(name)", {}, 'render_as_json() got null; give a default, e.g. `render_as_json(x :- "")`'],
  ['join(s, "-")', { s: "abc" }, "join() takes a list, got str"],
  ['join(xs, "-")', {}, "join() got null; give a default, e.g. `join(x :- [])`"],
  ['join(xs, "-")', { xs: ["a", null] }, "join() got a null element; drop or default it before joining"],
])("a builtin rejects null and wrong types: %s", (text, values, message) => {
  // A builtin never turns null or a wrong-typed value into text (`upper(null)` is not
  // "NONE"); it fails like arithmetic over null does.
  const name = text.slice(0, text.indexOf("("));
  expectMessage(text, `expression function '${name}' failed: ${message}`, fromRecord(values));
});

test("a builtin over null in a condition fails instead of routing", () => {
  const error = errorOf('upper(name) == "NONE"', undefined, ResolveMode.CONDITION_FALSY);
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message).toBe(
    "expression function 'upper' failed: upper() got null; give a default, e.g. `upper(x :- \"\")`",
  );
});

test("join stringifies numbers and render_as_json keeps a nested null", () => {
  expect(evaluate('join(xs, "-")', fromRecord({ xs: [1n, 2n] }))).toBe("1-2");
  expect(evaluate("render_as_json(d, 0)", fromRecord({ d: { k: null } }))).toBe('{\n"k": null\n}');
});

describe("the shipped builtins' TS-era rules", () => {
  const FILE = new FileRef({ uri: "s3://b/k" });

  test("join writes each element in the text form", () => {
    expect(evaluate('join([1, true, 1.5, "a"], "-")')).toBe("1-true-1.5-a");
    expect(evaluate('join(xs, "|")', fromRecord({ xs: [[1n, "a"], { k: null }, FILE] }))).toBe(
      '[1,"a"]|{"k":null}|{"uri":"s3://b/k","mime":null,"name":null}',
    );
  });

  test("render_as_json uses JS key order, the float text form, and keeps non-ASCII", () => {
    const d = { b: 1, a: [Infinity, null], "1": "é" };
    expect(evaluate("render_as_json(d, 2)", fromRecord({ d }))).toBe(
      '{\n  "1": "é",\n  "b": 1,\n  "a": [\n    Infinity,\n    null\n  ]\n}',
    );
  });

  test("render_as_json with indent 0 breaks lines without indenting", () => {
    expect(evaluate("render_as_json(x, 0)", fromRecord({ x: [1n, [2n], {}, []] }))).toBe("[\n1,\n[\n2\n],\n{},\n[]\n]");
  });

  test("render_as_json writes an empty list or record bare", () => {
    expect(evaluate("render_as_json([], 2)")).toBe("[]");
    expect(evaluate("render_as_json(d, 2)", fromRecord({ d: {} }))).toBe("{}");
  });

  test("render_as_json writes a file at depth as its mapping", () => {
    expect(evaluate("render_as_json(d)", fromRecord({ d: { f: FILE } }))).toBe(
      '{\n  "f": {\n    "uri": "s3://b/k",\n    "mime": null,\n    "name": null\n  }\n}',
    );
  });

  test("a big int renders its digits in both builtins", () => {
    const digits = "1" + "0".repeat(30);
    expect(evaluate('join([10 ** 30], "-")')).toBe(digits);
    expect(evaluate("render_as_json(10 ** 30)")).toBe(digits);
  });

  test("render_as_json quotes text", () => {
    expect(evaluate("render_as_json(s)", fromRecord({ s: 'a"b' }))).toBe('"a\\"b"');
  });

  test("render_as_json with indent 4 escapes as Python's json.dumps does", () => {
    // json.dumps({"s": "a\nb\\c\x01"}, indent=4, ensure_ascii=False)
    expect(evaluate("render_as_json(d, 4)", fromRecord({ d: { s: "a\nb\\c\u0001" } }))).toBe(
      '{\n    "s": "a\\nb\\\\c\\u0001"\n}',
    );
  });

  test.each([
    ["-1", "-1"],
    ['"  "', '"  "'],
    ["1.5", "1.5"],
    ["null", "null"],
    ["true", "true"],
  ])("render_as_json rejects the indent %s", (indent, shown) => {
    expectMessage(
      `render_as_json(1, ${indent})`,
      `expression function 'render_as_json' failed: render_as_json() takes an int indent of 0 or more, got ${shown}`,
    );
  });

  test("an indent too large for the runtime fails the same on every engine", () => {
    // V8 and JavaScriptCore both throw a RangeError from `repeat`, with different text.
    expectMessage(
      "render_as_json([1], 2 ** 40)",
      "expression function 'render_as_json' failed: text is too long for this runtime",
    );
  });

  test("join takes only text for sep", () => {
    const xs = fromRecord({ xs: ["a"] });
    expectMessage("join(xs, 1)", "expression function 'join' failed: join() takes str for sep, got int", xs);
    expectMessage("join(xs, null)", "expression function 'join' failed: join() takes str for sep, got null", xs);
  });

  test("upper names a file's kind", () => {
    expectMessage(
      "upper(f)",
      "expression function 'upper' failed: upper() takes str, got file",
      fromRecord({ f: FILE }),
    );
  });

  test("upper uses the full case mapping", () => {
    expect(evaluate('upper("ß")')).toBe("SS");
    expect(evaluate('lower("ÀB")')).toBe("àb");
  });
});

test("dotted access on a builtin call's result", () => {
  // The builtin returns a record; `.k` reads its field.
  standIn("identity", [{ name: "v" }], (v) => v);
  expect(evaluate("identity(d).k", fromRecord({ d: { k: "v" } }))).toBe("v");
});

test("an unknown builtin raises", () => {
  expectMessage("no_such_fn(x)", "unknown expression function 'no_such_fn'", fromRecord({ x: 1n }));
});

// --------------------------------------------------------------------------------------- //
// `item` scope (MAP-body-local)
// --------------------------------------------------------------------------------------- //

test("item scope", () => {
  expect(evaluate("item.title", undefined, { item: { title: "T" } })).toBe("T");
});

test("a bare item", () => {
  expect(evaluate("item", undefined, { item: { a: 1n } })).toEqual({ a: 1n });
});

test("item scope reads fields only", () => {
  expect(evaluate("item.__class__", undefined, { item: { a: 1n } })).toBeNull();
});

test("item-headed paths never reach resolve", () => {
  expect(evaluate("item.k", () => ({ used: 1, present: true, value: { k: "resolved" }, why: null }))).toBeNull();
});

// --------------------------------------------------------------------------------------- //
// `and` / `or` fold by truthiness
// --------------------------------------------------------------------------------------- //

test("or folds by truthiness", () => {
  expect(evaluate("x or y", fromRecord({ x: 0n, y: "hit" }))).toBe(true);
});

test("and folds by truthiness", () => {
  expect(evaluate("x and y", fromRecord({ x: "", y: "hit" }))).toBe(false);
});

// --------------------------------------------------------------------------------------- //
// Membership, unary minus on a miss, and list edge cases
// --------------------------------------------------------------------------------------- //

test("not in a list: true", () => {
  expect(evaluate('"z" not in items', fromRecord({ items: ["a", "b"] }))).toBe(true);
});

test("not in a list: false", () => {
  expect(evaluate('"a" not in items', fromRecord({ items: ["a", "b"] }))).toBe(false);
});

test("unary minus on a missing reference raises", () => {
  expect(errorOf("-missing", undefined, ResolveMode.BINDING_NONE)).toBeInstanceOf(ExpressionError);
});

test("a list literal with a missing element", () => {
  expect(evaluate("[a, 1]", undefined, { mode: ResolveMode.BINDING_NONE })).toEqual([null, 1n]);
});

// --------------------------------------------------------------------------------------- //
// Coalesce / default / required precedence, pinned against the grammar's grouping. Every
// bare name is a reference, so in `a:-b | c` the grouping must be `(a:-b) | c`, and in
// `a | b + 1` it must be `a | (b + 1)`: coalesce is the lowest, n-ary and flat.
// --------------------------------------------------------------------------------------- //

/** Return the name of a bare-reference operand (a `refcall` with one `NAME` child). */
function refName(node: ExprNode | undefined): string {
  expect(node instanceof Tree && node.data === "refcall" && node.children.length === 1).toBe(true);
  return String((node as Tree).children[0]);
}

test("default binds tighter than coalesce (tree)", () => {
  const tree = parseExpr("a:-b | c");
  expect(tree instanceof Tree && tree.data === "coalesce").toBe(true);
  const [first, second] = (tree as Tree).children;
  expect(first instanceof Tree && first.data === "default_expr").toBe(true);
  expect(refName((first as Tree).children[0])).toBe("a");
  expect(refName((first as Tree).children[1])).toBe("b");
  expect(refName(second)).toBe("c");
});

test("(a:-b) | c: the default supplies, then the coalesce short-circuits", () => {
  expect(evaluate("a:-b | c", fromRecord({ b: 10n, c: null }))).toBe(10n);
});

test("(a:-b) | c: falls through to c", () => {
  expect(evaluate("a:-b | c", fromRecord({ c: 20n }))).toBe(20n);
});

test("coalesce binds looser than arithmetic (tree)", () => {
  const tree = parseExpr("a | b + 1");
  expect(tree instanceof Tree && tree.data === "coalesce").toBe(true);
  const [first, second] = (tree as Tree).children;
  expect(refName(first)).toBe("a");
  expect(second instanceof Tree && second.data === "sum").toBe(true);
});

test("a | (b + 1): the arithmetic operand supplies", () => {
  expect(evaluate("a | b + 1", fromRecord({ b: 4n }))).toBe(5n);
});

test("a | (b + 1): the arithmetic arm short-circuits", () => {
  expect(evaluate("a | b + 1", fromRecord({ a: 7n }))).toBe(7n);
});

test("an n-ary coalesce is one flat node", () => {
  const tree = parseExpr("a | b | c");
  expect(tree instanceof Tree && tree.data === "coalesce").toBe(true);
  expect((tree as Tree).children.map(refName)).toEqual(["a", "b", "c"]);
});

test("every arm of an n-ary coalesce is reachable", () => {
  expect(evaluate("a | b | c", fromRecord({ c: 3n }))).toBe(3n);
  expect(evaluate("a | b | c", fromRecord({ b: 2n, c: 3n }))).toBe(2n);
  expect(evaluate("a | b | c", fromRecord({ a: 1n, b: 2n, c: 3n }))).toBe(1n);
});

// --------------------------------------------------------------------------------------- //
// TS-era rules
// --------------------------------------------------------------------------------------- //

describe("tokens", () => {
  test.each([
    ["1", 1n],
    ["1.0", 1],
    ["2.5", 2.5],
    ['"a b"', "a b"],
    ["'x'", "x"],
    ["true", true],
    ["false", false],
    ["null", null],
    ["none", null],
    ["inf", Infinity],
  ] as [string, Value][])("%s", (text, expected) => {
    expect(evaluate(text)).toBe(expected);
  });

  test("an unknown token or node is an error", () => {
    const resolve = MISSING_ROOT;
    expect(() => evalExpr(new Token("WEIRD", "x"), resolve)).toThrow("unexpected token WEIRD 'x'");
    expect(() => evalExpr(new Tree("weird", []), resolve)).toThrow("unsupported expression node 'weird'");
  });
});

describe("calls", () => {
  /** Register `f(a, b="B", c="C")`, returning its arguments as a list, and record each call's. */
  function recordingF(): Value[][] {
    const calls: Value[][] = [];
    standIn("f", [{ name: "a" }, { name: "b", default: "B" }, { name: "c", default: "C" }], (...args) => {
      calls.push(args);
      return args;
    });
    return calls;
  }

  test("keyword arguments reach their parameter and defaults fill in", () => {
    const calls = recordingF();
    expect(evaluate('f(1, c="z")')).toEqual([1n, "B", "z"]);
    expect(evaluate("f(b=2, a=1)")).toEqual([1n, 2n, "C"]);
    expect(calls).toEqual([
      [1n, "B", "z"],
      [1n, 2n, "C"],
    ]);
  });

  test("a builtin sees ints as bigints and floats as numbers", () => {
    const calls = recordingF();
    evaluate("f(1, 2.5, x)", fromRecord({ x: 10n ** 30n }));
    expect(calls).toEqual([[1n, 2.5, 10n ** 30n]]);
    expect(calls[0]!.map((arg) => typeof arg)).toEqual(["bigint", "number", "bigint"]);
  });

  test("a default written as a JS number reaches the builtin as an int", () => {
    const calls: Value[][] = [];
    standIn("pretty", [{ name: "value" }, { name: "indent", default: 2 }], (...args) => {
      calls.push(args);
      return "";
    });
    evaluate("pretty(1)");
    expect(calls).toEqual([[1n, 2n]]);
  });

  test("arguments that don't bind fail the call at evaluation", () => {
    recordingF();
    // `parseExpr` doesn't check calls (`parseCondition` does), so the binding fails here.
    expectMessage("f(1, 2, 3, 4)", "expression function 'f' failed: too many positional arguments");
    expectMessage("f(b=1)", "expression function 'f' failed: missing a required argument: 'a'");
    expectMessage("f(1, d=2)", "expression function 'f' failed: got an unexpected keyword argument 'd'");
    standIn("g", [{ name: "a" }], () => null);
    expectMessage("g(1, 2, 3)", "expression function 'g' failed: too many positional arguments");
  });

  test("an error thrown by a builtin names the builtin", () => {
    standIn("boom", [], () => {
      throw new TypeError("it broke");
    });
    expectMessage("boom()", "expression function 'boom' failed: it broke");
  });

  test("a thrown non-Error shows in its string form", () => {
    standIn("boom", [], () => {
      throw "plain text";
    });
    expectMessage("boom()", "expression function 'boom' failed: plain text");
    standIn("boom", [], () => {
      throw 42;
    });
    expectMessage("boom()", "expression function 'boom' failed: 42");
  });

  test("an ExpressionError thrown by a builtin passes through unchanged", () => {
    const thrown = new ExpressionError("upper() got null");
    standIn("boom", [], () => {
      throw thrown;
    });
    expect(errorOf("boom()")).toBe(thrown);
  });

  test("an argument's own error is not wrapped", () => {
    recordingF();
    expectMessage("f(1 / 0)", "division by zero: 1 / 0");
  });

  test.each([
    ["3 reads as 3n", 3, 3n],
    ["2.5 stays a float", 2.5, 2.5],
    ["text stays text", "text", "text"],
    ["a list reads item by item", [1, [2.5, null]], [1n, [2.5, null]]],
    ["a record reads field by field", { a: 1 }, { a: 1n }],
    ["undefined reads as null", undefined, null],
  ] as [string, unknown, Value][])("a result of %s", (_label, result, expected) => {
    standIn("f", [], () => result);
    expect(evaluate("f()")).toEqual(expected);
  });

  test.each([
    ["NaN", NaN, "result: NaN is not a yian value"],
    ["a nested undefined", [undefined], "result at [0]: undefined is not a yian value; use null"],
    ["a Date", new Date(0), "result: Date is not a yian value"],
  ] as [string, unknown, string][])("a result of %s fails the call", (_label, result, message) => {
    standIn("f", [], () => result);
    expectMessage("f()", `expression function 'f' failed: ${message}`);
  });

  test("a strict miss on a call result's field names the call", () => {
    standIn("identity", [{ name: "v" }], (v) => v);
    const error = errorOf("identity(d).nope.k", fromRecord({ d: { k: 1n } }), ResolveMode.STRICT_RAISE);
    expect(error).toBeInstanceOf(ExpressionError);
    expect(error.message).toBe("unresolved reference ${identity(...).nope.k} in expression");
    expect(evaluate("identity(d).nope", fromRecord({ d: { k: 1n } }))).toBeNull();
  });

  test("a strict miss in an argument raises", () => {
    standIn("identity", [{ name: "v" }], (v) => v);
    expect(errorOf("identity(d)", undefined, ResolveMode.STRICT_RAISE).message).toBe(
      "unresolved reference ${d} in expression",
    );
    // In the non-strict modes the builtin gets null.
    expect(evaluate("identity(d)")).toBeNull();
  });
});

describe("number model", () => {
  test.each([
    ["2 ** 200", 2n ** 200n],
    ["10 ** 30 + 1", 10n ** 30n + 1n],
    ["true + 1", 2n],
    ["true + 1.5", 2.5],
    ["-true", -1n],
    ["3 * 2 + 0.5", 6.5],
    ["10 ** 400 * 1.0", Infinity],
  ] as [string, Value][])("%s", (text, expected) => {
    expect(evaluate(text)).toBe(expected);
  });
});

describe("/ gives a correctly rounded float", () => {
  test.each([
    ["8 / 2", 4],
    ["7 / 2", 3.5],
    ["0 / -5", -0],
    // Beyond 2^53; the expected doubles are Python's `a / b`.
    ["(2 ** 53 + 1) / 1", 9007199254740992],
    ["(2 ** 54 + 2) / 2", 9007199254740992],
    ["(2 ** 54 + 3) / 2", 9007199254740994],
    ["10 ** 400 / 10 ** 399", 10],
    ["1 / 10 ** 400", 0],
    ["3 / 2 ** 1075", 1e-323],
    ["1 / 2 ** 1074", 5e-324],
    ["1 / 2 ** 1075", 0],
    ["(10 ** 30 + 7) / 3", 3.333333333333333e29],
    ["-(10 ** 25) / (10 ** 24 + 1)", -10],
    ["(2 ** 1024 - 2 ** 970 - 1) / 1", 1.7976931348623157e308],
    // Python raises an overflow here; TS gives Infinity.
    ["10 ** 309 / 1", Infinity],
    ["-(10 ** 400) / 7", -Infinity],
    ["(2 ** 1024 - 2 ** 970) / 1", Infinity],
  ] as [string, number][])("%s", (text, expected) => {
    expect(evaluate(text)).toBe(expected);
  });

  test("1000 seeded scaled pairs divide as their unscaled doubles", () => {
    const random = mulberry32(53);
    const draw53 = (): bigint => {
      const magnitude = (BigInt(random() >>> 11) << 32n) | BigInt(random());
      return random() & 1 ? -magnitude : magnitude;
    };
    const failures: string[] = [];
    for (let i = 0; i < 1000; i++) {
      const x = draw53();
      let y = draw53();
      while (y === 0n) y = draw53();
      const k = 60n + BigInt(random() % 141);
      const scale = 2n ** k;
      const got = evaluate("a / b", fromRecord({ a: x * scale, b: y * scale }));
      if (!Object.is(got, Number(x) / Number(y))) failures.push(`${x} / ${y} at 2 ** ${k}: ${String(got)}`);
    }
    expect(failures).toEqual([]);
  });
});

describe("% takes the sign of the left", () => {
  test.each([
    ["-7 % 2", -1n],
    ["7 % -2", 1n],
    ["-7.5 % 2", -1.5],
    ["5 % inf", 5],
  ] as [string, Value][])("%s", (text, expected) => {
    expect(evaluate(text)).toBe(expected);
  });
});

describe("**", () => {
  // The expected doubles are Python's.
  test.each([
    ["2 ** -1", 0.5],
    ["10 ** -4", 0.0001],
    ["(-2) ** -3", -0.125],
    ["3 ** -40", 8.225263339969959e-20],
    ["2 ** -1074", 5e-324],
    ["2 ** -1101", 0],
    ["(-2) ** -1101", -0],
    ["(10 ** 400) ** -1", 0],
    ["(2 ** 53 + 1) ** -1", 1.1102230246251565e-16],
    ["1.1 ** 10", 2.5937424601000023],
    ["10.0 ** -5", 1e-5],
    ["1.05 ** 30", 4.321942375150668],
    ["0 ** 0", 1n],
    ["1 ** (10 ** 30)", 1n],
    ["(-1) ** (10 ** 30 + 1)", -1n],
    ["1 ** inf", 1],
    ["(-1) ** inf", 1],
    ["(-1) ** -inf", 1],
    ["2.0 ** 5000", Infinity],
    ["inf ** 0", 1],
  ] as [string, Value][])("%s", (text, expected) => {
    expect(evaluate(text)).toBe(expected);
  });

  test.each([
    ["x ** 1", 5e-324],
    ["x ** -1", Infinity],
  ])("%s with the subnormal x = 5e-324", (text, expected) => {
    expect(evaluate(text, fromRecord({ x: 5e-324 }))).toBe(expected);
  });
});

describe("division by zero", () => {
  test.each([
    ["1 / 0", "division by zero: 1 / 0"],
    ["1 % 0", "division by zero: 1 % 0"],
    ["1.5 / 0.0", "division by zero: 1.5 / 0"],
    ["1 / (0.0 * -1)", "division by zero: 1 / 0"],
    ["1 / false", "division by zero: 1 / false"],
    ["0 ** -1", "division by zero: 0 ** -1"],
    ["0.0 ** -0.5", "division by zero: 0 ** -0.5"],
    ["0 ** -inf", "division by zero: 0 ** -Infinity"],
  ])("%s", (text, message) => {
    expectMessage(text, message);
  });
});

describe("a NaN result is an error", () => {
  test.each([
    ["inf - inf", "- on Infinity and Infinity has no numeric result"],
    ["0 * inf", "* on 0 and Infinity has no numeric result"],
    ["inf / inf", "/ on Infinity and Infinity has no numeric result"],
    ["inf % 2", "% on Infinity and 2 has no numeric result"],
    ["(0 - 8) ** 0.5", "** on -8 and 0.5 has no numeric result"],
    ["10 ** 400 - inf", `- on ${BIG_SHOWN} and Infinity has no numeric result`],
  ])("%s", (text, message) => {
    expectMessage(text, message);
  });

  test("an infinite result is a value", () => {
    expect(evaluate("inf + 1")).toBe(Infinity);
    expect(evaluate("-inf")).toBe(-Infinity);
    expect(evaluate("x * 10", fromRecord({ x: 1e308 }))).toBe(Infinity);
  });
});

describe("removed operators", () => {
  test.each([
    ['"%s" % 1', "text formatting with `%` was removed; use a template"],
    ['"ab" * 3', 'type error evaluating "ab" * 3: unsupported operand kinds for `*`: `str` and `int`'],
    ["[0] * 3", "type error evaluating [0] * 3: unsupported operand kinds for `*`: `list` and `int`"],
    ['3 * "ab"', 'type error evaluating 3 * "ab": unsupported operand kinds for `*`: `int` and `str`'],
    ['"a" + 1', 'type error evaluating "a" + 1: unsupported operand kinds for `+`: `str` and `int`'],
    ['[1] + "a"', 'type error evaluating [1] + "a": unsupported operand kinds for `+`: `list` and `str`'],
    ["r + s", 'type error evaluating {"k":1} + {"j":2}: unsupported operand kinds for `+`: `record` and `record`'],
    ["x + 1", "type error evaluating null + 1: unsupported operand kinds for `+`: `null` and `int`"],
    ['-"a"', 'unary minus on non-number "a": unsupported operand kind for `-`: `str`'],
    ["-xs", "unary minus on non-number [1]: unsupported operand kind for `-`: `list`"],
  ])("%s", (text, message) => {
    expectMessage(text, message, fromRecord({ r: { k: 1n }, s: { j: 2n }, xs: [1n] }));
  });
});

test("an integer result the runtime refuses", () => {
  expectMessage("2 ** (2 ** 40)", "integer result is too large for this runtime");
});

describe("comparisons", () => {
  const resolve = fromRecord({ r: { a: 1n, b: 2n }, s: { b: 2n, a: 1n }, t: { a: 1n } });
  test.each([
    ["1 == 1.0", true],
    ["2 ** 60 == 2.0 ** 60", true],
    ["2 ** 53 + 1 == 2.0 ** 53", false],
    ["true == 1", true],
    ["true == 1.0", true],
    ["inf == inf", true],
    ["1 < inf", true],
    ["10 ** 400 < inf", true],
    ["[1, [2]] == [1, [2]]", true],
    ["r == s", true],
    ["r == t", false],
    ["r != t", true],
    ["null == null", true],
    ["null == 0", false],
    ['"1" == 1', false],
    ["[1] == [1.0]", true],
    ['"a" < "b"', true],
    ['"Z" < "a"', true],
    ['"￿" < "\u{1F600}"', false],
    ["x < 1", false],
    ["true > 0", true],
    ["2 >= 2.0", true],
    ["3 <= 2", false],
  ] as [string, boolean][])("%s", (text, expected) => {
    expect(evaluate(text, resolve)).toBe(expected);
  });

  test.each([
    ["[1] < [2]", "type error evaluating [1] < [2]: unsupported operand kinds for `<`: `list` and `list`"],
    ['"a" < 1', 'type error evaluating "a" < 1: unsupported operand kinds for `<`: `str` and `int`'],
  ])("%s is an error", (text, message) => {
    expectMessage(text, message);
  });
});

describe("membership", () => {
  const resolve = fromRecord({ r: { k: 1n } });
  test.each([
    ['"b" in "abc"', true],
    ["1 in [1.0]", true],
    ["[1] in [[1]]", true],
    ['"k" in r', true],
    ["1 in r", false],
    ['"toString" in r', false],
    ['"d" not in "abc"', true],
  ] as [string, boolean][])("%s", (text, expected) => {
    expect(evaluate(text, resolve)).toBe(expected);
  });

  test.each([
    ['1 in "abc"', 'type error evaluating 1 in "abc": unsupported operand kinds for `in`: `int` and `str`'],
    ['1 not in "abc"', 'type error evaluating 1 not in "abc": unsupported operand kinds for `not in`: `int` and `str`'],
    ["1 in x", "type error evaluating 1 in null: unsupported operand kinds for `in`: `int` and `null`"],
    ['"a" in 3', 'type error evaluating "a" in 3: unsupported operand kinds for `in`: `str` and `int`'],
  ])("%s is an error", (text, message) => {
    expectMessage(text, message);
  });
});

describe("truthiness, and, or, not", () => {
  const resolve = fromRecord({ empty: {}, full: { k: 1n } });
  test.each([
    ["not 0.0", true],
    ["not 0", true],
    ['not ""', true],
    ["not []", true],
    ["not empty", true],
    ["not full", false],
    ["not inf", false],
    ["not null", true],
    ['"a" or 0', true],
    ["0 and x", false],
    ["1 and [0]", true],
  ] as [string, boolean][])("%s", (text, expected) => {
    expect(evaluate(text, resolve)).toBe(expected);
  });
});

describe("expressions at the nesting limit evaluate and walk", () => {
  const resolve = fromRecord({ x: 1n });

  test("100 nested lists", () => {
    const text = "[".repeat(100) + "x" + "]".repeat(100);
    let expected: Value = 1n;
    for (let i = 0; i < 100; i++) expected = [expected];
    expect(evaluate(text, resolve)).toEqual(expected);
    expect(exprRefs(parseExpr(text))).toEqual(["x"]);
    expect(rewriteExprRefs(text, () => ({ text: "y", used: 1 }))).toBe("[".repeat(100) + "y" + "]".repeat(100));
  });

  test("100 nested unary minuses", () => {
    const text = "-(".repeat(100) + "x" + ")".repeat(100);
    expect(evaluate(text, resolve)).toBe(1n);
    expect(exprRefs(parseExpr(text))).toEqual(["x"]);
    expect(rewriteExprRefs(text, () => ({ text: "y", used: 1 }))).toBe("-(".repeat(100) + "y" + ")".repeat(100));
  });

  test("100 nested calls", () => {
    standIn("identity", [{ name: "v" }], (v) => v);
    for (const open of ["identity(", "identity(v="]) {
      const text = open.repeat(100) + "x" + ")".repeat(100);
      expect(evaluate(text, resolve)).toBe(1n);
      expect(exprRefs(parseExpr(text))).toEqual(["x"]);
      expect(rewriteExprRefs(text, () => ({ text: "y", used: 1 }))).toBe(open.repeat(100) + "y" + ")".repeat(100));
    }
  });

  test("100 levels of not, minus and comparisons", () => {
    let text = "x";
    let expected: bigint | boolean = 1n;
    for (let i = 0; i < 100; i++) {
      if (i % 3 === 0) {
        text = `-(${text})`;
        expected = -BigInt(expected);
      } else if (i % 3 === 1) {
        text = `(${text}) < 0`;
        expected = BigInt(expected) < 0n;
      } else {
        text = `not (${text})`;
        expected = !expected;
      }
    }
    expect(evaluate(text, resolve)).toBe(expected);
    expect(exprRefs(parseExpr(text))).toEqual(["x"]);
    expect(rewriteExprRefs(text, () => ({ text: "y", used: 1 }))).toBe(text.replace("x", "y"));
  });
});

/** A small seeded PRNG giving 32-bit unsigned ints. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}

test("a comprehension with 5,000 if clauses evaluates without overflowing the stack", () => {
  expect(evaluate(`[x for x in xs${" if true".repeat(5000)}]`, fromRecord({ xs: [1n, 2n] }))).toEqual([
    1n,
    2n,
  ]);
});

test("?. reads like . on today's resolvers", () => {
  const resolve = fromRecord({ a: { k: 1n }, n: null });
  expect(evaluate("a?.k", resolve)).toBe(1n);
  expect(evaluate("n?.k", resolve)).toBeNull();
  expect(evaluate("[m?.k for m in xs]", fromRecord({ xs: [{ k: 1n }, null] }))).toEqual([1n, null]);
});

test("a strict miss names the read as written", () => {
  expect(errorOf("a?.b.c", undefined, ResolveMode.STRICT_RAISE).message).toBe(
    "unresolved reference ${a?.b.c} in expression",
  );
});

test("a resolver returns a root read; the evaluator walks the steps", () => {
  const seen: string[][] = [];
  const resolve = (names: readonly string[]) => {
    seen.push([...names]);
    return { used: 1, present: true, value: { k: { m: 1n } }, why: null };
  };
  expect(evalExpr(parseExpr("a.k.m"), resolve)).toBe(1n);
  expect(seen).toEqual([["a", "k", "m"]]);
});

// --- declared types --- //

const str = Type.scalar(ValueKind.STRING);
const Rec = new Type({ kind: ValueKind.OBJECT, fields: new Map([["k", str], ["opt", str]]), required: new Set(["k"]) });
const A = new Type({ kind: ValueKind.OBJECT, fields: new Map([["x", str]]), required: new Set() });
const B = new Type({ kind: ValueKind.OBJECT, fields: new Map([["y", str]]), required: new Set() });
const U = new Type({ kind: ValueKind.OBJECT, variants: [A, B] });

test("declaredAt projects a type through declared fields", () => {
  expect(typeEquals(declaredAt(Rec, ["opt"])!, str)).toBe(true);
  expect(declaredAt(Rec, ["nope"])).toBeNull();
  expect(declaredAt(Type.scalar(ValueKind.ANY), ["k"])).toBeNull();
  expect(declaredAt(Type.scalar(ValueKind.OBJECT), ["k"])).toBeNull();
  expect(typeEquals(declaredAt(U, ["x"])!, str)).toBe(true); // only A declares x
  expect(typeEquals(declaredAt(U, [])!, U)).toBe(true);
  expect(typeEquals(declaredAt(Rec.replace({ nullable: true }), [])!, Rec)).toBe(true);
  expect(declaredAt(null, [])).toBeNull();
});

test("localDeclared reads a local's type, then its fields", () => {
  expect(typeEquals(localDeclared(new Map([["r", Rec]]))(["r", "opt"])!, str)).toBe(true);
  expect(localDeclared(new Map([["r", Rec]]))(["s"])).toBeNull();
  expect(localDeclared(null)(["r"])).toBeNull();
});

// --- the step table (field reads) --- //

/** A root reader over `record` whose absent names say why. */
const rec = (record: ValueRecord): RootResolver => (names) => {
  const [present, value] = fieldOf(record, names[0]!);
  return { used: 1, present, value, why: present ? null : `input '${names[0]}' is not bound` };
};
const ev = (text: string, record: ValueRecord, mode: ResolveMode = ResolveMode.BINDING_NONE) =>
  evalExpr(parseExpr(text), rec(record), { mode });

test.each([
  ["n.k", { n: null }, "cannot read ${n.k}: n is null; write n?.k to read null instead, or check n != null first"],
  ["gone.k", {}, "cannot read ${gone.k}: gone is missing (input 'gone' is not bound)"],
  ["u.k", { u: {} }, "cannot read ${u.k}: u has no field 'k'"],
  ["t.k", { t: "text" }, "cannot read ${t.k}: t is str, which has no fields"],
  [
    "f.k",
    { f: new FileRef({ uri: "file:///x" }) },
    "cannot read ${f.k}: f is a file, which has only uri, mime and name",
  ],
  [
    "a.b.c",
    { a: { b: null } },
    "cannot read ${a.b.c}: a.b is null; write a.b?.c to read null instead, or check a.b != null first",
  ],
])(". raises: %s", (text, record, message) => {
  expect(() => ev(text, record as ValueRecord)).toThrow(message);
});

test.each([
  ["n?.k", { n: null }, null],
  ["gone?.k", {}, null],
  ["u?.k", { u: {} }, null],
  ["n?.k.m", { n: null }, null],
  ["n", { n: null }, null],
  ["gone :- 'd'", {}, "d"],
  ["n?.k :- 'd'", { n: null }, "d"],
  ["f.mime", { f: new FileRef({ uri: "file:///x" }) }, null],
])("value: %s", (text, record, value) => {
  expect(ev(text, record as ValueRecord)).toEqual(value);
});

test("?. still raises on a non-record", () => {
  expect(() => ev("t?.k", { t: "text" })).toThrow("cannot read ${t?.k}: t is str, which has no fields");
});

test(":- and :? don't catch a field-read error", () => {
  expect(() => ev("n.k :- 'd'", { n: null })).toThrow("n is null");
  expect(() => ev('n.k :? "need k"', { n: null })).toThrow("n is null");
  expect(() => ev('gone :? "need it"', {})).toThrow("need it");
  expect(() => ev("n.k | 'd'", { n: null })).toThrow("n is null");
});

test("a?.b.c: when a exists and a.b is null, .c raises", () => {
  expect(() => ev("a?.b.c", { a: { b: null } })).toThrow("cannot read ${a?.b.c}: a?.b is null");
});

test("a bare missing root keeps today's per-mode handling", () => {
  expect(ev("gone", {})).toBeNull();
  expect(ev("gone", {}, ResolveMode.CONDITION_FALSY)).toBeNull();
  expect(() => ev("gone", {}, ResolveMode.STRICT_RAISE)).toThrow("unresolved reference ${gone} in expression");
});

test("comprehension variables, item and call results follow the same table", () => {
  expect(() => ev("[x.k for x in xs]", { xs: [null] })).toThrow("cannot read ${x.k}: x is null");
  expect(ev("[x?.k for x in xs]", { xs: [null, { k: 1n }] })).toEqual([null, 1n]);
  expect(() => evalExpr(parseExpr("item.k"), MISSING_ROOT, { item: null })).toThrow(
    "cannot read ${item.k}: item is null",
  );
  expect(evalExpr(parseExpr("item?.k"), MISSING_ROOT, { item: null })).toBeNull();
  expect(() => ev("upper(t).k", { t: "a" })).toThrow("cannot read ${upper(...).k}: upper(...) is str");
});

test("a variable the loader could not narrow names the variable and the field it lacks", () => {
  // `or`-combined isinstance doesn't narrow at load (#1175); the read raises at run time.
  const T = new Type({ kind: ValueKind.OBJECT, fields: new Map([["text", str]]), required: new Set(["text"]) });
  const N = new Type({ kind: ValueKind.OBJECT, fields: new Map([["n", str]]), required: new Set(["n"]) });
  const types = new Map([["T", T], ["N", N]]);
  const text = "[p.text for p in ps if isinstance(p, T) or isinstance(p, N)]";
  expect(() => evalExpr(parseExpr(text), rec({ ps: [{ text: "a" }, { n: "b" }] }), { types })).toThrow(
    "cannot read ${p.text}: p has no field 'text'",
  );
});
