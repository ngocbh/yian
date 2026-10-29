/**
 * `T(...)` constructors in the expression language: call shape, the positional and keyword forms,
 * larger expressions, names that are not constructors, a constructor in an embedded template span,
 * the reference walkers, co-skip, flow callees and prompts, the binder (a failed constructor is
 * `binding.type`, a good one delivers a record), and static inference.
 */

import { expect, test } from "bun:test";

import {
  ConstructorTypeError,
  constructorNames,
  ExpressionError,
  evalExpr,
  evaluateWhenRecord,
  exprRefs,
  inferExprType,
  parseCondition,
  type RootResolver,
  rewriteExprRefs,
} from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { routeFeedback, SourceRef } from "../../src/expr/lineage.ts";
import {
  bindingCoSkips,
  evalBinding,
  flowCallCalleesInSpans,
  promptRefs,
  spanCallDirectiveText,
} from "../../src/expr/template.ts";
import { BindingError, bindParams } from "../../src/nodes/binding.ts";
import { Parameter, ValueTemplate } from "../../src/nodes/contracts.ts";
import { Type, typeFor, type Value, type ValueRecord, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { fieldOf } from "../../src/typesys/values.ts";

const REGISTRY = readTypedefs({
  Buy: { action: "Literal(buy)", conviction: "float" },
  Sell: { action: "Literal(sell)", price: "float" },
  Decision: "Union(Buy, Sell)",
  Size: "Literal(s, m, l)",
});
const TYPES = new Map(["Buy", "Sell", "Decision", "Size"].map((name) => [name, typeFor(name, REGISTRY)]));

const A_BUY: ValueRecord = { action: "buy", conviction: 0.5 };
const A_SELL: ValueRecord = { action: "sell", price: 1.5 };

/** Return a `resolve` over `values`: the head, then `fieldOf` per dotted step; a miss is `null`. */
function resolver(values: ValueRecord = {}): RootResolver {
  return (names) => {
    const [present, value] = fieldOf(values, names[0]!);
    return { used: 1, present, value, why: null };
  };
}

/** Evaluate `text` against `values` with `TYPES`. */
function evaluate(text: string, values: ValueRecord = {}, types: ReadonlyMap<string, Type> = TYPES): Value {
  return evalExpr(parseExpr(text), resolver(values), { types });
}

/** Return the error `run` throws, failing if it returns. */
function errorOf(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Expect `run` to throw a `cls` whose message is exactly `message`. */
function expectError(run: () => unknown, message: string, cls: typeof ExpressionError = ExpressionError): void {
  const error = errorOf(run);
  expect(error).toBeInstanceOf(cls);
  expect(error.message).toBe(message);
}

// --- call shape --- //

test("a constructor takes keywords or one value", () => {
  for (const text of ["Buy(a, b)", 'Buy(a, action="buy")', 'Buy(action="buy", a)']) {
    // The callee isn't known to be a constructor until it resolves, so each of these parses.
    expect(() => parseExpr(text)).not.toThrow();
    expectError(
      () => evaluate(text, { a: A_BUY, b: A_BUY }),
      "Buy(...) takes keyword fields, as in Buy(name=value, ...), or one value to check, as in Buy(x)",
    );
  }
});

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

test("a call repeating a keyword is rejected", () => {
  const cases: [string, string][] = [
    ["Buy(action=1, action=2)", "Buy(...) repeats the keyword 'action'"],
    ['join(xs, sep=",", sep=";")', "join(...) repeats the keyword 'sep'"],
  ];
  for (const [text, message] of cases) expectMessage(text, message);
});

test("a builtin still mixes positional and keyword arguments", () => {
  expect(evaluate('join(xs, sep="-")', { xs: ["a", "b"] })).toBe("a-b");
});

// --- positional form: check and narrow --- //

test("a value that fits is given back unchanged", () => {
  expect(evaluate("Buy(d.output)", { d: { output: A_BUY } })).toEqual(A_BUY);
});

test("an int checked as a float field stays an int", () => {
  const value = evaluate("Buy(d.output)", { d: { output: { action: "buy", conviction: 3n } } }) as ValueRecord;
  expect(value.conviction).toBe(3n);
});

test("a value missing a required field fails naming it", () => {
  expectError(
    () => evaluate("Buy(d.output)", { d: { output: { conviction: 0.5 } } }),
    'Buy(...): record missing required fields: ["action"]',
    ConstructorTypeError,
  );
});

test("a missing value fails the check", () => {
  // A missing root is null in a binding; a field of it would raise before the check.
  expectError(() => evaluate("Buy(d)"), "Buy(...): null is not an object for a record type", ConstructorTypeError);
  expectError(() => evaluate("Buy(d.output)"), "cannot read ${d.output}: d is missing");
});

test("a union typedef checks against every member", () => {
  expect(evaluate("Decision(d.output)", { d: { output: A_SELL } })).toEqual(A_SELL);
  expectError(
    () => evaluate("Decision(d.output)", { d: { output: { action: "hold" } } }),
    `Decision(...): "hold" is not a tag of the union (expected one of ["buy","sell"])`,
    ConstructorTypeError,
  );
});

test("an enum alias checks a scalar", () => {
  expect(evaluate("Size(x)", { x: "m" })).toBe("m");
  expectError(
    () => evaluate("Size(x)", { x: "xl" }),
    `Size(...): "xl" is not one of Literal(l, m, s)`,
    ConstructorTypeError,
  );
});

test("a field read on the result", () => {
  expect(evaluate("Buy(d.output).conviction", { d: { output: A_BUY } })).toBe(0.5);
});

test("a constructor error is an expression error", () => {
  expect(new ConstructorTypeError("x")).toBeInstanceOf(ExpressionError);
});

// --- keyword form: build a record --- //

test("keywords build a record from exactly the given fields", () => {
  const value = evaluate('Buy(action="buy", conviction=s.output)', { s: { output: 0.7 } });
  expect(value).toEqual({ action: "buy", conviction: 0.7 });
});

test("a built record missing a field fails naming it", () => {
  expectError(
    () => evaluate("Buy(conviction=s.output)", { s: { output: 0.7 } }),
    'Buy(...): record missing required fields: ["action"]',
    ConstructorTypeError,
  );
});

test("a built record with a wrong field fails", () => {
  expectError(
    () => evaluate('Buy(action="buy", conviction="high")'),
    'Buy(...): "high" does not match declared type float',
    ConstructorTypeError,
  );
});

test("a keyword named __proto__ is a field, never the prototype", () => {
  const types = new Map([["Bag", new Type({ kind: ValueKind.OBJECT })]]);
  const value = evaluate("Bag(__proto__=1)", {}, types) as ValueRecord;
  expect(Object.hasOwn(value, "__proto__")).toBe(true);
  expect(Object.getPrototypeOf(value)).toBe(Object.prototype);
});

// --- inside larger expressions --- //

test("constructors in a list literal", () => {
  const value = evaluate("[Buy(a.output), Sell(b.output)]", { a: { output: A_BUY }, b: { output: A_SELL } });
  expect(value).toEqual([A_BUY, A_SELL]);
});

test("a constructor in a condition", () => {
  const record = { d: A_BUY };
  expect(evaluateWhenRecord("Buy(d).conviction > 0.1", record, TYPES)).toBe(true);
  expectError(
    () => evaluateWhenRecord("Sell(d).price > 0.1", record, TYPES),
    'Sell(...): record missing required fields: ["price"]',
    ConstructorTypeError,
  );
});

test("a constructor in an embedded template span", () => {
  const text = evalBinding("conviction ${Buy(d.output).conviction}", resolver({ d: { output: A_BUY } }), {
    types: TYPES,
  });
  expect(text).toBe("conviction 0.5");
});

// --- names that are not constructors --- //

/** Expect `parseCondition(text, TYPES)` to fail with exactly `message`. */
function expectConditionMessage(text: string, message: string): void {
  let error: unknown;
  try {
    parseCondition(text, TYPES);
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe(message);
}

test("a builtin type name is not a conversion", () => {
  for (const name of ["int", "str", "list", "Optional"]) {
    const message =
      `unknown expression function '${name}'; types in expressions are typedef names; ` +
      `\`${name}(...)\` is not a conversion`;
    expectConditionMessage(`${name}(x)`, message);
    // Even with the name in the type table, evaluation never treats it as a constructor.
    const types = new Map([...TYPES, [name, Type.scalar(ValueKind.INTEGER)]]);
    expectError(() => evaluate(`${name}(x)`, { x: 1n }, types), message);
  }
});

test("a name absent from the type table is an unknown function", () => {
  expectConditionMessage("Hold(x)", "unknown expression function 'Hold'");
  expectError(() => evaluate("Hold(x)", { x: 1n }), "unknown expression function 'Hold'");
});

test("reference walks read every argument and skip the names", () => {
  const tree = parseExpr('Buy(action="buy", conviction=s.output) == Buy(d.output)');
  expect(exprRefs(tree)).toEqual(["s.output", "d.output"]);
});

test("rewrites leave the type and field names alone", () => {
  const renamed = rewriteExprRefs("Buy(action=a, conviction=s.output).action", (path) => ({
    text: "x." + path,
    used: path.split(".").length,
  }));
  expect(renamed).toBe("Buy(action=x.a, conviction=x.s.output).action");
});

test("a positional constructor co-skips like its argument", () => {
  expect(bindingCoSkips("${Buy(d.output)}")).toBe(true);
  expect(bindingCoSkips("${Buy(d.output).conviction}")).toBe(true);
  expect(bindingCoSkips('${Buy(action="buy", conviction=d.output)}')).toBe(false);
  expect(bindingCoSkips("${upper(d.output)}")).toBe(false);
});

test("constructor names are the callees that are no builtin", () => {
  const tree = parseExpr("[Buy(d.output), upper(x), isinstance(y, Sell), Sell(price=p), Buy(z)]");
  expect(constructorNames(tree)).toEqual(["Buy", "Sell"]);
  expect(constructorNames(parseExpr("x"))).toEqual([]);
});

test("a typedef name is not a flow callee", () => {
  const text = "${Buy(d.output)} ${review(d.output)}";
  expect(flowCallCalleesInSpans(text)).toEqual(["Buy", "review"]);
  expect(flowCallCalleesInSpans(text, new Set(["Buy"]))).toEqual(["review"]);
});

test.each([
  ["a call in text", "see ${call(enrich, topic=input.topic)}", "call(enrich, topic=input.topic)"],
  ["the first span", "${call(a)} ${call(b)}", "call(a)"],
  ["the outer of nested calls", "${call(outer, x=call(inner))}", "call(outer, x=call(inner))"],
  ["a call inside an expression", "${upper(call (my-flow))}", "call (my-flow)"],
  ["no call", "${review(x)}", null],
])("the source of a span call(...): %s", (_name, text, source) => {
  expect(spanCallDirectiveText(text)).toBe(source);
});

test("a constructor is not a prompt function", () => {
  expectError(
    () => promptRefs("Order: ${Buy(order)}"),
    "unknown prompt function 'Buy'; prompts call only formatting builtins " +
      "(join, lower, render_as_json, upper); build or check a typed value with a " +
      "`typedefs:` constructor `T(...)` in an input binding",
  );
});

test("a builtin type name in a prompt is not a conversion", () => {
  expectError(
    () => promptRefs("Count: ${int(order)}"),
    "unknown prompt function 'int'; types in expressions are typedef names; `int(...)` is not a conversion",
  );
});

test("a prompt call with several arguments is an unknown prompt function", () => {
  const error = errorOf(() => promptRefs("${foo(a, b)}"));
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message.startsWith("unknown prompt function 'foo';")).toBe(true);
});

// --- lineage --- //

/** Return a traced `resolve` over `values`: each head is sourced at its name in run 1. */
function tracedResolver(values: ValueRecord): RootResolver {
  const resolve = resolver(values);
  return (names) => ({ ...resolve(names), source: new SourceRef("run", 1, [names[0]!]) });
}

test("a positional constructor routes to its argument", () => {
  const traced = evalBinding("${Buy(d.output)}", tracedResolver({ d: { output: A_BUY } }), {
    trace: true,
    types: TYPES,
  });
  expect(traced.value).toEqual(A_BUY);
  const route = traced.route;
  expect(route.kind).toBe("selected");
  expect(route.expression).toBe("Buy(d.output)");
  expect(route.children[0]!.source).toEqual(new SourceRef("run", 1, ["d", "output"]));
});

test("a keyword constructor routes each field", () => {
  const traced = evalBinding('${Buy(action="buy", conviction=s.output)}', tracedResolver({ s: { output: 0.7 } }), {
    trace: true,
    types: TYPES,
  });
  const route = traced.route;
  expect(route.kind).toBe("record");
  expect(route.names).toEqual(["action", "conviction"]);
  expect(route.expression).toBe('Buy(action="buy", conviction=s.output)');
  expect(route.children[1]!.source).toEqual(new SourceRef("run", 1, ["s", "output"]));
});

test("a field read on a constructor projects its route", () => {
  const traced = evalBinding("${Buy(d.output).conviction}", tracedResolver({ d: { output: A_BUY } }), {
    trace: true,
    types: TYPES,
  });
  expect(traced.value).toBe(0.5);
  expect(traced.route.kind).toBe("project");
  expect(traced.route.path).toEqual(["conviction"]);
});

// --- the binder --- //

/** Return a pool holding each entry of `outputs` as that node's output. */
function poolOf(outputs: Record<string, unknown>): VariablePool {
  const pool = new VariablePool();
  for (const [nodeId, value] of Object.entries(outputs)) pool.set(nodeId, value);
  return pool;
}

test("the binder reports a failed constructor as binding.type", () => {
  const error = errorOf(() =>
    bindParams([new Parameter("x")], new Map([["x", "${Buy(d.output)}"]]), poolOf({ d: { conviction: 0.5 } }), {
      types: TYPES,
    }),
  );
  expect(error).toBeInstanceOf(BindingError);
  expect((error as BindingError).failureCode).toBe("binding.type");
  expect(error.message).toContain('record missing required fields: ["action"]');
});

test("the binder reports a failed nested constructor as binding.type", () => {
  const error = errorOf(() =>
    bindParams(
      [new Parameter("x")],
      new Map([["x", new ValueTemplate({ d: "${Buy(d.output)}" })]]),
      poolOf({ d: { conviction: 0.5 } }),
      { types: TYPES },
    ),
  );
  expect(error).toBeInstanceOf(BindingError);
  expect((error as BindingError).failureCode).toBe("binding.type");
  expect(error.message).toContain("binding 'x.d'");
});

test("the binder delivers a constructed record", () => {
  const record = bindParams(
    [new Parameter("x")],
    new Map([["x", '${Buy(action="buy", conviction=d.output)}']]),
    poolOf({ d: 0.25 }),
    { types: TYPES },
  );
  expect(record).toEqual({ x: { action: "buy", conviction: 0.25 } });
});

// --- static inference --- //

/** Return the type a constructor name names, else `null`. */
function typeFn(name: string): Type | null {
  return TYPES.get(name) ?? null;
}

/** Resolve no reference. */
function noRefs(): Type | null {
  return null;
}

test("inference gives a constructor its type", () => {
  expect(inferExprType(parseExpr("Buy(d.output)"), noRefs, typeFn)).toBe(TYPES.get("Buy")!);
  expect(inferExprType(parseExpr('Buy(action="buy", conviction=1)'), noRefs, typeFn)).toBe(TYPES.get("Buy")!);
});

test("inference reads fields of a constructor result", () => {
  expect(inferExprType(parseExpr("Buy(d.output).conviction"), noRefs, typeFn)).toEqual(
    Type.scalar(ValueKind.NUMBER),
  );
  expect(inferExprType(parseExpr("Buy(d.output).size"), noRefs, typeFn)).toBeNull();
});

test("inference reads a field every union member has", () => {
  const inferred = inferExprType(parseExpr("Decision(d.output).action"), noRefs, typeFn);
  expect(inferred).toEqual(new Type({ kind: ValueKind.STRING, tags: new Set(["buy", "sell"]) }));
});

test("inference reads optional record fields, file fields and nullable aliases", () => {
  const order = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["qty", Type.scalar(ValueKind.INTEGER)],
      ["note", Type.scalar(ValueKind.STRING)],
    ]),
    required: new Set(["qty"]),
  });
  const types = new Map([
    ["Order", order],
    ["Doc", Type.scalar(ValueKind.FILE)],
    ["MaybeBuy", TYPES.get("Buy")!.replace({ nullable: true })],
  ]);
  const infer = (text: string) => inferExprType(parseExpr(text), noRefs, (name) => types.get(name) ?? null);
  const str = Type.scalar(ValueKind.STRING);
  const optionalStr = str.replace({ nullable: true });
  expect(infer("Order(x).qty")).toEqual(Type.scalar(ValueKind.INTEGER));
  // A field the record doesn't require reads as Optional.
  expect(infer("Order(x).note")).toEqual(optionalStr);
  expect(infer("Doc(x).uri")).toEqual(str);
  expect(infer("Doc(x).mime")).toEqual(optionalStr);
  expect(infer("Doc(x).name")).toEqual(optionalStr);
  expect(infer("Doc(x).size")).toBeNull();
  // A read through a nullable alias reads as Optional.
  expect(infer("MaybeBuy(x).conviction")).toEqual(Type.scalar(ValueKind.NUMBER).replace({ nullable: true }));
  expect(infer("MaybeBuy(x).action")).toEqual(
    new Type({ kind: ValueKind.STRING, tags: new Set(["buy"]), nullable: true }),
  );
});

test("inference without types leaves a call unknown", () => {
  expect(inferExprType(parseExpr("Buy(d.output)"), noRefs)).toBeNull();
});

test("feedback on a built field reaches its argument", () => {
  const sources: Record<number, ValueRecord> = { 1: { output: 0.7 }, 2: { output: A_BUY } };
  const resolve: RootResolver = (names) => {
    const seq = names[0] === "s" ? 1 : 2;
    return { used: 1, present: true, value: sources[seq]!, why: null, source: new SourceRef("run", seq) };
  };
  const lookup = (source: SourceRef) => sources[source.executionSeq];
  const built = evalBinding('${Buy(action="buy", conviction=s.output)}', resolve, { trace: true, types: TYPES });
  let routed = routeFeedback(built.route, { conviction: "too high" }, lookup);
  expect(routed.contributions.map((c) => [c.source, c.feedback])).toEqual([
    [new SourceRef("run", 1), { output: "too high" }],
  ]);
  const checked = evalBinding("${Buy(d.output)}", resolve, { trace: true, types: TYPES });
  routed = routeFeedback(checked.route, { conviction: "too high" }, lookup);
  expect(routed.contributions.map((c) => [c.source, c.feedback])).toEqual([
    [new SourceRef("run", 2), { output: { conviction: "too high" } }],
  ]);
});
