/**
 * A `${...}` marks an expression inside a YAML string; it can't appear inside one: the
 * `parseExpr` and condition tests of `tests/engine/test_nested_span_rejected.py`, which check
 * that a whole-span condition reads as its bare spelling, and the error a `${...}` nested inside
 * an expression raises and the rewrite it shows, plus the template-span tests (`scanTemplate`).
 *
 * The harvested `load-core/engine_nested_span_rejected__case_condition_shows_the_rewrite`, from
 * `test_case_condition_with_a_nested_span_fails_to_load_with_the_rewrite`, is pinned at the end
 * (`assemble`, whose CASE build raises it), and so is
 * `load-core/engine_nested_span_rejected__loop_predicate_shows_the_rewrite` (`assembledWith`, the
 * `step` def built by hand; the LOOP build raises it), and
 * `load-core/engine_nested_span_rejected__binding_shows_the_rewrite` (the load half of
 * `test_binding_with_a_nested_span_fails_to_load_with_the_rewrite`; `assemble`, whose inline-call
 * desugar raises it). `test_node_assert_with_a_nested_span_fails_to_load_with_the_rewrite` is
 * ported through `assembleText`, pinning the whole message, and its case
 * `load-core/engine_nested_span_rejected__node_assert_shows_the_rewrite` (`assemble`, whose
 * node-assert check raises it) is pinned at the end.
 *
 * Later (T3-20, `lowerFlowAsserts`): `test_flow_assert_with_a_nested_span_fails_to_load_with_the_rewrite`
 * and its case.
 *
 * Left for later: the case, loop and prompt load tests and the run tests land with compose and
 * `run.ts`.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import {
  conditionRefs,
  ExpressionError,
  parseCondition,
  rewriteConditionRefs,
} from "../../src/expr/expressions.ts";
import { type ExprNode, parseExpr, rejectNestedSpans, Token } from "../../src/expr/grammar.ts";
import { scanTemplate } from "../../src/expr/template.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

const NESTED = "`${...}` can't appear inside an expression";

/** A position-blind form of `node`, as Lark's `Tree`/`Token` equality ignores positions. */
function plain(node: ExprNode): unknown {
  if (node instanceof Token) return { type: node.type, value: node.value };
  return { data: node.data, children: node.children.map(plain) };
}

// --- whole-span conditions: one `${...}` around the whole value --- //

test("whole-span condition matches its bare spelling", () => {
  expect(plain(parseCondition("${a > 5}"))).toEqual(plain(parseCondition("a > 5")));
});

test("whole-span condition with a brace in a string literal", () => {
  expect(conditionRefs('${x == "}"}')).toEqual(["x"]);
});

test("whole-span rewrite with a brace in a string literal", () => {
  const rename = (path: string) => (path === "x" ? "y" : null);
  expect(rewriteConditionRefs('${x == "}"}', rename)).toBe('${y == "}"}');
});

// --- a `${...}` inside an expression is an error that shows the rewrite --- //

/** Return the error `fn` throws, which must be an `ExpressionError`. */
function caught(fn: () => unknown): Error {
  let error: unknown;
  try {
    fn();
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  return error as Error;
}

/** Expect `parseExpr(text)` to fail with the nested-span error and no rewrite. */
function expectNoRewrite(text: string): void {
  expect(caught(() => parseExpr(text)).message).toBe(NESTED);
}

test("nested span shows the rewrite", () => {
  const cases: [string, string][] = [
    ["input.a :- ${input.b}", "input.a :- input.b"],
    ["${a.output} > 5", "a.output > 5"],
    ["${a.output:-0} >= 1", "(a.output:-0) >= 1"],
    ["x :- ${ y }", "x :- y"],
    ["x :- ${y + 1}", "x :- (y + 1)"],
    ["a :- ${b :- ${c}}", "a :- (b :- c)"],
    ["isinstance(x, ${Buy})", "isinstance(x, Buy)"],
    ["render_as_json(${briefs}, 4)", "render_as_json(briefs, 4)"],
    ["not${x}", "not x"],
    ["x :- ${and}", "x :- and"],
  ];
  for (const [text, rewrite] of cases) {
    expect(caught(() => parseExpr(text)).message).toBe(`${NESTED}; write \`${rewrite}\``);
  }
});

test("nested span without a parsable rewrite shows none", () => {
  for (const text of ["a :- ${b", "a :- ${b +}", "${}"]) expectNoRewrite(text);
});

test("a span inside a string literal is text", () => {
  for (const text of ['x == "${y}"', "x == '${y}'"]) expect(() => parseExpr(text)).not.toThrow();
});

test("a whole-span condition holding a nested span shows the bare rewrite", () => {
  expect(caught(() => parseCondition("${a > ${b}}")).message).toBe(`${NESTED}; write \`a > b\``);
});

test("a template span shows the whole fixed span", () => {
  const cases: [string, string][] = [
    ["${input.as_of:-${system.today}}", "${input.as_of:-system.today}"],
    ["pre ${upper(${name})} post", "${upper(name)}"],
  ];
  for (const [template, rewrite] of cases) {
    expect(caught(() => scanTemplate(template)).message).toBe(`${NESTED}; write \`${rewrite}\``);
  }
});

test("a template span with a quoted span is text", () => {
  expect(() => scanTemplate('${x == "${y}"}')).not.toThrow();
});

const DEEP_SPANS = "${".repeat(3000) + "x" + "}".repeat(3000);

test("deeply nested spans are rejected without a rewrite", () => {
  expectNoRewrite(DEEP_SPANS);
});

test("deeply nested template spans are rejected without a rewrite", () => {
  expect(caught(() => scanTemplate(DEEP_SPANS)).message).toBe(NESTED);
});

test("a rewrite past the nesting limit is not suggested", () => {
  // The rewrite `not (not ... true)` would be 101 levels deep, so it would not parse.
  expectNoRewrite("not ${" + "not ".repeat(100) + "true}");
});

test("a rewrite at the nesting limit is suggested", () => {
  expect(caught(() => parseExpr("not ${" + "not ".repeat(99) + "true}")).message).toBe(
    `${NESTED}; write \`not (${"not ".repeat(99)}true)\``,
  );
});

test("a span interior's rewrite is the whole fixed span", () => {
  expect(caught(() => rejectNestedSpans("a :- ${b}", { inSpan: true })).message).toBe(
    `${NESTED}; write \`\${a :- b}\``,
  );
});

test("the rewrite only needs to parse, not pass the later checks", () => {
  expect(caught(() => rejectNestedSpans("${a.b(x)}")).message).toBe(`${NESTED}; write \`(a.b(x))\``);
  expect(caught(() => rejectNestedSpans("${isinstance(x)}")).message).toBe(
    `${NESTED}; write \`(isinstance(x))\``,
  );
});

// --- assert loading --- //

const ASSERTS = `
id: f
name: f
input:
  n: !int
nodes:
  n:
    kind: code
    code: "return inputs['n']"
    input: {n: '\${input.n}'}
    output: int
    asserts:
      - <node_assert>
output: \${n.output}
asserts:
  - <flow_assert>
`;

test("a node assert with a nested span fails to load with the rewrite", () => {
  const text = ASSERTS.replace("<node_assert>", "${output} < 0").replace("<flow_assert>", "input.n > 0");
  expect(loadError(() => assembleText(text)).message).toBe(
    "flow has invalid node asserts:\n" +
      `  node 'n' assert '\${output} < 0': not a valid boolean expression (${NESTED}; write \`output < 0\`)`,
  );
});

// ---------- harvested conformance cases ----------

test("assembled: load-core/engine_nested_span_rejected__case_condition_shows_the_rewrite", () => {
  runCase({ id: "load-core/engine_nested_span_rejected__case_condition_shows_the_rewrite", call: assembled });
});

/**
 * The `step` def of the LOOP case: it reads `{n, exited}`, and its producer `s` returns the record
 * `{n: int, exited: bool}`, which the def outputs.
 */
function stepBody(): ComposedFlow {
  const record = new ValueContract(typeFor("Rec", readTypedefs({ Rec: { n: "int", exited: "bool" } })));
  return handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n  exited: !bool\n", new Map()),
    producers: { s: record },
    output: "${s.output}",
  });
}

test("assembled: load-core/engine_nested_span_rejected__loop_predicate_shows_the_rewrite", () => {
  runCase({
    id: "load-core/engine_nested_span_rejected__loop_predicate_shows_the_rewrite",
    call: assembledWith({ step: stepBody() }),
  });
});

test("assembled: load-core/engine_nested_span_rejected__binding_shows_the_rewrite", () => {
  runCase({ id: "load-core/engine_nested_span_rejected__binding_shows_the_rewrite", call: assembled });
});

test("assembled: load-core/engine_nested_span_rejected__node_assert_shows_the_rewrite", () => {
  runCase({ id: "load-core/engine_nested_span_rejected__node_assert_shows_the_rewrite", call: assembled });
});
