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
 * desugar raises it).
 *
 * The load rows `test_binding_with_a_nested_span_fails_to_load_with_the_rewrite`,
 * `test_case_condition_with_a_nested_span_fails_to_load_with_the_rewrite`,
 * `test_flow_assert_with_a_nested_span_fails_to_load_with_the_rewrite`,
 * `test_node_assert_with_a_nested_span_fails_to_load_with_the_rewrite`,
 * `test_loop_predicate_with_a_nested_span_fails_to_load_with_the_rewrite` (its `step` def built by
 * hand, `stepBody`) and `test_prompt_with_a_nested_span_fails_to_load_with_the_rewrite` load with
 * `assembleText` (`assemble` as the loader calls it) where Python calls `load_flow`, each checking
 * that its error has a range, as Python's `_load_error` does, the assert rows pinning the whole
 * message; their cases `load-core/engine_nested_span_rejected__{flow,node}_assert_shows_the_rewrite`
 * (`assemble`, whose flow-assert lowering and node-assert check raise them) and
 * `load-core/engine_nested_span_rejected__prompt_shows_the_rewrite` (the reference validator) are
 * pinned at the end.
 *
 * Later (T4-8 follow-up): the run tests (`test_parentheses_and_coalesce_cover_a_default_chain`,
 * `test_documented_case_conditions_route`, `test_case_on_a_whole_span_routes` and
 * `test_loop_predicate_in_the_bare_spelling_runs`), with their cases
 * `run-core/engine_nested_span_rejected__{a_default_chain,case_on_a_whole_span_routes,
 * documented_case_condition_routes}`.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
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
  type ChildFlows,
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

const BINDING = `
id: f
name: f
input:
  a: !Optional(str) null
  b: !Optional(str) null
nodes:
  n:
    kind: code
    code: "return inputs['v']"
    input:
      v: \${input.a :- \${input.b}}
    output: Optional(str)
output: \${n.output}
`;

/** Return the `LoadError` assembling `text` raises, checking that it is located (has a range). */
function loadFails(text: string, children: ChildFlows = {}): LoadError {
  const error = loadError(() => assembleText(text, "flow.yaml", { children }));
  expect(error.diagnostic.frames.at(-1)?.range).toBeDefined();
  return error;
}

test("a binding with a nested span fails to load with the rewrite", () => {
  expect(loadFails(BINDING).message).toContain("write `${input.a :- input.b}`");
});

/** The CASE flow whose `gate` routes on `when`. */
const searched = (when: string) => `
id: f
name: f
input:
  n: !Optional(int) null
  approve: !bool false
  label: !str none
nodes:
  score:
    kind: code
    code: "return inputs['n']"
    input: {n: '\${input.n}'}
    output: Optional(int)
  gate:
    kind: case
    cases:
      - when: ${when}
        then: hot
    else: cold
  hot: {kind: code, code: "return 'hot'", output: str}
  cold: {kind: code, code: "return 'cold'", output: str}
output: \${hot.output | cold.output}
`;

test.each([
  ['"${score.output} > 5"', "score.output > 5"],
  ['"${score.output:-0} >= 1"', "(score.output:-0) >= 1"],
])("a case condition with a nested span fails to load with the rewrite: %s", (when, rewrite) => {
  expect(loadFails(searched(when)).message).toContain(`write \`${rewrite}\``);
});

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

test("a flow assert with a nested span fails to load with the rewrite", () => {
  const text = ASSERTS.replace("<node_assert>", "output >= 0").replace("<flow_assert>", "${input.n} > 0");
  expect(loadFails(text).message).toBe(
    `assert '\${input.n} > 0' is not a valid boolean expression: ${NESTED}; write \`input.n > 0\``,
  );
});

test("a node assert with a nested span fails to load with the rewrite", () => {
  const text = ASSERTS.replace("<node_assert>", "${output} < 0").replace("<flow_assert>", "input.n > 0");
  expect(loadFails(text).message).toBe(
    "flow has invalid node asserts:\n" +
      `  node 'n' assert '\${output} < 0': not a valid boolean expression (${NESTED}; write \`output < 0\`)`,
  );
});

const LOOP = `
id: f
name: f
defs:
  step:
    input:
      n: !int
      exited: !bool
    nodes:
      s:
        kind: code
        code: "return {'n': inputs['n'] + 1, 'exited': inputs['n'] >= 2}"
        input:
          n: \${input.n}
        output:
          n: int
          exited: bool
    output: \${s.output}
nodes:
  repeat:
    kind: loop
    call: step
    input:
      n: 0
      exited: false
    while: "not \${exited}"
    max: 5
output: \${repeat.output}
`;

test("a loop predicate with a nested span fails to load with the rewrite", () => {
  expect(loadFails(LOOP, { step: stepBody() }).message).toContain("write `not exited`");
});

const PROMPT = `
id: f
name: f
input:
  name: !str
nodes:
  brief:
    kind: agent
    input: {name: '\${input.name}'}
    output: str
    prompt: "Hello \${upper(\${name})}"
output: \${brief.output}
`;

test("a prompt with a nested span fails to load with the rewrite", () => {
  expect(loadFails(PROMPT).message).toContain("write `${upper(name)}`");
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

test("assembled: load-core/engine_nested_span_rejected__prompt_shows_the_rewrite", () => {
  runCase({ id: "load-core/engine_nested_span_rejected__prompt_shows_the_rewrite", call: assembled });
});

test("assembled: load-core/engine_nested_span_rejected__node_assert_shows_the_rewrite", () => {
  runCase({ id: "load-core/engine_nested_span_rejected__node_assert_shows_the_rewrite", call: assembled });
});

test("assembled: load-core/engine_nested_span_rejected__flow_assert_shows_the_rewrite", () => {
  runCase({ id: "load-core/engine_nested_span_rejected__flow_assert_shows_the_rewrite", call: assembled });
});
