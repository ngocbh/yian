/**
 * The whole-value `call(<flow-id>, kw=...)` compile-time directive.
 *
 * An unquoted value whose trimmed text starts with `call(` must be exactly one directive, else it
 * is a load error naming the problem; a quoted value is text. The first arg is the POSITIONAL flow
 * id, the rest are `name=value` keywords. It desugars through `desugarInlineCalls` and a shared
 * minter into a synth `CallDescriptor` (`over = null`), the host binding rewritten to
 * `${<synth>.output}`. The rows drive the pass directly with a deterministic minter; the load rows
 * assemble the flow (`assembleText`), a `defs:` callee built by hand (`handChild`) where the build
 * resolves it.
 */

import { afterEach, expect, test } from "bun:test";

import { INLINE_FLOW_PREFIX } from "../../src/compose/desugar/inline_bodies.ts";
import { ArgumentValueError, desugarInlineCalls, synthIdGen } from "../../src/compose/desugar/inline_calls.ts";
import { CALL_HEAD, matchCallDirective } from "../../src/compose/helpers/directives.ts";
import { LoadDiagnostics, type LoadError } from "../../src/compose/helpers/errors.ts";
import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { parseYamlLiteral } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import { findParenEnd, InlineCall, splitCallsAware } from "../../src/expr/template.ts";
import { Assertion } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import {
  assembled,
  assembleText,
  handChild,
  loadConformanceFlow,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

afterEach(() => forget());

test.each([
  ["call(summarize, messages=${messages})", ["summarize", " messages=${messages}"]],
  ["call(f)", ["f", ""]],
  ["  call (f, x=1)  ", ["f", " x=1"]],
  ["call(my-flow, a=1)", ["my-flow", " a=1"]],
  ["call(f, 30)", ["f", " 30"]],
] as const)("a whole-value directive %p is recognized", (value, expected) => {
  expect(matchCallDirective(value)).toEqual([...expected]);
});

test.each(["call-me(x)", "pre call(x)", "recall(x)", "plain text"])(
  "%p is not a directive",
  (value) => {
    expect(matchCallDirective(value)).toBeNull();
  },
);

test("CALL_HEAD anchors at the start", () => {
  expect(CALL_HEAD.test("call (f)")).toBe(true);
  expect(CALL_HEAD.test("pre call(f)")).toBe(false);
});

/** Assert `matchCallDirective(value)` throws an `ExpressionError` with exactly `message`. */
function expectThrows(value: string, message: string): void {
  let caught: unknown;
  try {
    matchCallDirective(value);
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(ExpressionError);
  expect((caught as ExpressionError).message).toBe(message);
}

const NEVER_CLOSED = "`call(` is never closed (check its parentheses and quotes)";
const notFirst = (first: string) =>
  `a call(...) must start with a flow id (a bare name) and a comma, got ${first}`;

test.each([
  ["call(f) tail", "text follows the closing `)`: 'tail'; a call(...) must be the whole value"],
  ["call(f, x=1", NEVER_CLOSED],
  ['call(f, x="a)', NEVER_CLOSED],
  ["call(f, x=${g)", NEVER_CLOSED],
  ["call(f x=1)", notFirst("'f x=1'")],
  ["call(f.g)", notFirst("'f.g'")],
  ["call()", notFirst("''")],
  [
    "call(f) and more",
    "text follows the closing `)`: 'and more'; a call(...) must be the whole value",
  ],
])("a malformed directive %p throws its message", (value, message) => {
  expectThrows(value, message);
});

// The outer call keeps a malformed nested call in its remainder; the error comes from the nested
// call when the desugarer recurses into it.
test("a nested call's trailing text is the nested call's error", () => {
  expect(matchCallDirective("call(f, x=call(g) tail)")).toEqual(["f", " x=call(g) tail"]);
  expectThrows(
    "call(g) tail",
    "text follows the closing `)`: 'tail'; a call(...) must be the whole value",
  );
});

test.each([
  ["call(f, x=${g(1)})", 17],
  ['call(f, x=")")', 13],
  ["call(f, x=call(g, y=1))", 22],
  ["call(f)", 6],
  ["call(f, x=${g(1)", null],
  ["call(f, x=${g)", null],
  ["call(f, x=(1)", null],
] as const)("findParenEnd(%p) is %p", (text, expected) => {
  expect(findParenEnd(text, 5)).toBe(expected);
});

test.each([
  [
    'a=1, b=f(1, 2), c="x,y", d=${h(1, 2)}',
    ",",
    ["a=1", " b=f(1, 2)", ' c="x,y"', " d=${h(1, 2)}"],
  ],
  ["a=${[1, 2]}, b=1", ",", ["a=${[1, 2]}", " b=1"]],
  ["x=a=b", "=", ["x", "a", "b"]],
  ["a), b", ",", ["a)", " b"]],
  ["c='a=b'", "=", ["c", "'a=b'"]],
  ["", ",", [""]],
] as const)("splitCallsAware(%p, %p) splits at top level only", (text, sep, expected) => {
  expect(splitCallsAware(text, sep)).toEqual([...expected]);
});

test("an InlineCall is frozen and keeps its fields", () => {
  const call = new InlineCall("__call_0", "f", { a: 1n });
  expect(Object.isFrozen(call)).toBe(true);
  expect(call.id).toBe("__call_0");
  expect(call.callee).toBe("f");
  expect(call.args).toEqual({ a: 1n });
});

// ---------- desugarInlineCalls ----------

/** A minimal CODE descriptor whose `inputs` carry the binding under test. */
function code(nodeId: string, inputs: Record<string, unknown>): CodeDescriptor {
  return new CodeDescriptor({ id: nodeId, code: "mod:fn", inputs, outputs: "str" });
}

/** Descriptors keyed by id, in the given order. */
function nodes(...descriptors: NodeDescriptor[]): Map<string, NodeDescriptor> {
  return new Map(descriptors.map((desc) => [desc.id, desc]));
}

/** Return synth `callId` of `descriptors` as the `CallDescriptor` it is. */
function synthOf(descriptors: ReadonlyMap<string, NodeDescriptor>, callId: string): CallDescriptor {
  const desc = descriptors.get(callId);
  expect(desc).toBeInstanceOf(CallDescriptor);
  return desc as CallDescriptor;
}

/** The ids of the synth nodes in `descriptors`. */
const synthIds = (descriptors: ReadonlyMap<string, NodeDescriptor>) =>
  [...descriptors.keys()].filter((nodeId) => nodeId.startsWith("__call_"));

/** Desugar `use`'s `topic` binding `value`, expecting a load error, and return it. */
const topicError = (value: string, options: Parameters<typeof desugarInlineCalls>[2] = {}): LoadError =>
  loadError(() => desugarInlineCalls(nodes(code("use", { topic: value })), null, { nextId: synthIdGen(), ...options }));

test("whole value call directive recognized", () => {
  const [descriptors] = desugarInlineCalls(
    nodes(code("use", { topic: "call(summarize, messages=${messages})" })),
    null,
    { nextId: synthIdGen() },
  );
  expect(descriptors.get("use")!.inputs["topic"]).toBe("${__call_0.output}");
  const synth = synthOf(descriptors, "__call_0");
  expect(synth.call).toBe("summarize");
  expect(synth.inputs).toEqual({ messages: "${messages}" });
  expect(synth.over).toBeNull();
});

test("mid value call is literal not directive", () => {
  const [descriptors] = desugarInlineCalls(nodes(code("use", { topic: "pre call(x)" })), null, { nextId: synthIdGen() });
  expect(descriptors.get("use")!.inputs["topic"]).toBe("pre call(x)");
  expect(synthIds(descriptors)).toEqual([]);
});

const TRAILING_TAIL = "text follows the closing `)`: 'tail'; a call(...) must be the whole value";
const POSITIONAL = (flowId: string, arg: string) =>
  `call(${flowId}, ...) passes a positional argument '${arg}'; every argument after the flow id must be a ` +
  "keyword (name=value)";

test.each([
  ["call(f) tail", TRAILING_TAIL, '`"call(f) tail"`'],
  ["call(f, x=1", NEVER_CLOSED, '`"call(f, x=1"`'],
  ['call(f, x="a)', NEVER_CLOSED, "`'call(f, x=\"a)'`"],
  ["call(f x=1)", notFirst("'f x=1'"), '`"call(f x=1)"`'],
  ["call(f.g)", notFirst("'f.g'"), '`"call(f.g)"`'],
  ["call()", notFirst("''"), '`"call()"`'],
  ["call(f, 30)", POSITIONAL("f", "30"), '`"call(f, 30)"`'],
  [
    "call(f, a=1 b=2)",
    "call(f, ...) keyword 'a' value '1 b=2' looks like two keywords; add the missing comma, or quote the value",
    '`"call(f, a=1 b=2)"`',
  ],
  ["call(f, x=call(g) tail)", TRAILING_TAIL, '`"call(f, x=call(g) tail)"`'],
  ["call(f, x=call(g, 1))", POSITIONAL("g", "1"), '`"call(f, x=call(g, 1))"`'],
])("malformed directive %p is a load error with the quote escape", (value, problem, quoted) => {
  // An unquoted value starting with `call(` must be exactly one well-formed directive.
  expect(topicError(value).message).toBe(`use: ${problem}. If you meant the text, quote the whole value: ${quoted}`);
});

test("quoted directive is text", () => {
  // A quoted binding is text, so quoting is the escape for a literal `call(...)`.
  const [descriptors] = desugarInlineCalls(
    nodes(code("use", { topic: "call(f, x=1", note: "call(f, x=1)" })),
    null,
    { bindingStyleAt: () => "single", nextId: synthIdGen() },
  );
  expect(descriptors.get("use")!.inputs).toEqual({ topic: "call(f, x=1", note: "call(f, x=1)" });
  expect(synthIds(descriptors)).toEqual([]);
});

test("quoted directive holding a span is loud", () => {
  // Quoting makes a `call(...)` text, which can't be what a span-bearing call meant.
  const error = topicError("call(f, x=${input.x})", { bindingStyleAt: () => "double" });
  expect(error.message).toBe(
    "use: a quoted `call(...)` is text, so it can't hold `${...}` here; drop the quotes to call the flow, " +
      "or reword the text",
  );
});

test("quoted nested arg is text", () => {
  const [descriptors] = desugarInlineCalls(
    nodes(code("use", { topic: "call(f, x=\"call(g\", y='call(h)')" })),
    null,
    { nextId: synthIdGen() },
  );
  expect(synthOf(descriptors, "__call_0").inputs).toEqual({ x: "call(g", y: "call(h)" });
});

test("quoted flow output is text and unquoted is strict", () => {
  const [, outputs] = desugarInlineCalls(new Map(), { a: "call(f", b: "call(g)" }, {
    outputStyleAt: (key) => (key === "a" ? "double" : null),
    nextId: synthIdGen(),
  });
  expect(outputs).toEqual({ a: "call(f", b: "${__call_0.output}" });
  const error = loadError(() => desugarInlineCalls(new Map(), "call(f", { nextId: synthIdGen() }));
  expect(error.message).toBe(
    'flow outputs: `call(` is never closed (check its parentheses and quotes). If you meant the text, quote ' +
      'the whole value: `"call(f"`',
  );
});

test("malformed directive in assert is loud without a quote hint", () => {
  // An assert is an expression; quoting it does not make it text, so no quote hint.
  const error = loadError(() =>
    desugarInlineCalls(new Map(), null, { assertsSection: ["call(f) tail"], nextId: synthIdGen() }),
  );
  expect(error.message).toBe("assert: text follows the closing `)`: 'tail'; a call(...) must be the whole value");
});

test("bare id with hyphen recognized", () => {
  const [descriptors] = desugarInlineCalls(
    nodes(code("use", { topic: "call(my-flow, topic=${input.t})" })),
    null,
    { nextId: synthIdGen() },
  );
  expect(descriptors.get("use")!.inputs["topic"]).toBe("${__call_0.output}");
  expect(synthOf(descriptors, "__call_0").call).toBe("my-flow");
});

test("positional arg after flow id is loud", () => {
  expect(topicError("call(f, 30)").message).toBe(
    `use: ${POSITIONAL("f", "30")}. If you meant the text, quote the whole value: \`"call(f, 30)"\``,
  );
});

test("repeated keyword is loud", () => {
  expect(topicError("call(f, a=1, a=2)").message).toBe(
    "use: call(f, ...) passes keyword 'a' more than once. If you meant the text, quote the whole value: " +
      '`"call(f, a=1, a=2)"`',
  );
});

test("repeated keyword in nested directive is loud", () => {
  expect(topicError("call(f, x=call(g, b=1, b=2))").message).toContain(
    "call(g, ...) passes keyword 'b' more than once",
  );
});

test("directive capturing item is loud", () => {
  expect(topicError("call(wrap, t=${item})").message).toBe(
    "node 'use': inline call 'wrap' cannot capture ${item} (arg 't') — it desugars to a top-level call node " +
      "with no map-element scope; use a named `call` node (whose per-element `input:` may read `${item}`)",
  );
});

test("nested directive desugars inner first", () => {
  const [descriptors] = desugarInlineCalls(nodes(code("use", { topic: "call(a, x=call(b))" })), null, {
    nextId: synthIdGen(),
  });
  // Inner `b` is minted first (inner-first), then outer `a`.
  const inner = synthOf(descriptors, "__call_0");
  const outer = synthOf(descriptors, "__call_1");
  expect(inner.call).toBe("b");
  expect(outer.call).toBe("a");
  expect(outer.inputs["x"]).toBe("${__call_0.output}");
  expect(descriptors.get("use")!.inputs["topic"]).toBe("${__call_1.output}");
});

/** A flow whose CODE node `use` binds `topic` to `binding`. */
function privateFlow(flowId: string, binding: string): string {
  return `id: ${flowId}
name: ${flowId}
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: ${binding}
    output: str
output: \${use.output}
`;
}

// The test scope's resolver throws for any callee, so a `LoadError` here never reached child
// resolution.
test.each([
  ["outer", "private_outer", "d", (name: string) => `call(${name})`],
  ["nested", "private_nested", "a", (name: string) => `call(outer, x=call(${name}))`],
] as const)("%s private callee is rejected at binding before resolution", (_kind, flowId, letter, binding) => {
  const privateName = INLINE_FLOW_PREFIX + letter.repeat(64);
  const text = privateFlow(flowId, binding(privateName));

  const error = loadError(() => assembleText(text));

  const expectedLine = text.split("\n").findIndex((line) => line.includes(privateName)) + 1;
  expect(error.diagnostic.frames[0]!.range.startLine).toBe(expectedLine);
  expect(error.diagnostic.frames[0]!.text).toBe(text);
  expect(error.message).toContain(INLINE_FLOW_PREFIX);
  expect(error.message).not.toContain(privateName);
});

test.each([
  ["output", 17],
  ["assert", 23],
] as const)("private callee in flow section %p preserves section line", (section, expectedLine) => {
  const privateName = INLINE_FLOW_PREFIX + "b".repeat(64);
  const expectedRange = new SourceRange(expectedLine, 1, expectedLine, 1);
  const diagnostics = new LoadDiagnostics("flow.yaml", (" ".repeat(80) + "\n").repeat(30));
  const error = loadError(() =>
    section === "output"
      ? desugarInlineCalls(new Map(), `call(${privateName})`, {
          outputsRange: expectedRange,
          nextId: synthIdGen(),
          diagnostics,
        })
      : desugarInlineCalls(new Map(), null, {
          assertsSection: [`call(${privateName})`],
          assertionRangeAt: () => expectedRange,
          nextId: synthIdGen(),
          diagnostics,
        }),
  );

  expect(error.diagnostic.frames[0]!.range.startLine).toBe(expectedLine);
  expect(error.diagnostic.frames[0]!.label).toBe("flow.yaml");
  expect(error.message).toContain(INLINE_FLOW_PREFIX);
  expect(error.message).not.toContain(privateName);
});

test.each(["__inline_flow", "x__inline_flow_"])("private callee guard matches only the exact prefix: %p", (callee) => {
  const [descriptors] = desugarInlineCalls(nodes(code("use", { topic: `call(${callee})` })), null, {
    nextId: synthIdGen(),
  });

  expect(synthOf(descriptors, "__call_0").call).toBe(callee);
});

test("shared minter yields distinct ids across bindings", () => {
  const [descriptors] = desugarInlineCalls(
    nodes(code("a", { topic: "call(f, x=${input.x})" }), code("b", { topic: "call(g, y=${input.y})" })),
    null,
    { nextId: synthIdGen() },
  );
  expect(synthIds(descriptors)).toEqual(["__call_0", "__call_1"]); // distinct, no collision
});

test("old form rejected, new form desugars", () => {
  // The old `${flow(args)}` form is a hard-break load error; the `call(...)` form desugars.
  expect(() =>
    desugarInlineCalls(nodes(code("old", { topic: "${ enrich(topic=input.topic) }" })), null, { nextId: synthIdGen() }),
  ).toThrow(
    "old: a flow call 'enrich' inside ${...} is not supported — write it as the whole value with the call(...) " +
      "directive: call(enrich, ...)",
  );
  const [descriptors] = desugarInlineCalls(
    nodes(code("new", { topic: "call(summarize, messages=${messages})" })),
    null,
    { nextId: synthIdGen() },
  );
  expect(descriptors.get("new")!.inputs["topic"]).toBe("${__call_0.output}");
  expect(synthIds(descriptors).map((callId) => synthOf(descriptors, callId).call)).toEqual(["summarize"]);
});

test("no arg directive recognized", () => {
  const [descriptors] = desugarInlineCalls(nodes(code("use", { topic: "call(now)" })), null, { nextId: synthIdGen() });
  expect(descriptors.get("use")!.inputs["topic"]).toBe("${__call_0.output}");
  const synth = synthOf(descriptors, "__call_0");
  expect(synth.call).toBe("now");
  expect(synth.inputs).toEqual({});
});

/** Desugar `directive` as a node binding and return its synth call's inputs. */
function synthInputs(directive: string): Readonly<Record<string, unknown>> {
  const [descriptors] = desugarInlineCalls(nodes(code("use", { topic: directive })), null, { nextId: synthIdGen() });
  return synthOf(descriptors, "__call_0").inputs;
}

test("keyword literal values coerce like the named form", () => {
  // A bare literal kwarg binds its typed value (int/bool/null); a quoted scalar unwraps.
  expect(synthInputs('call(f, n=30, flag=true, z=null, s="hi")')).toEqual({ n: 30n, flag: true, z: null, s: "hi" });
});

test("bare bool word arg stays string", () => {
  // `yes` is text in both forms: the flow reads plain scalars with YAML 1.2 rules.
  expect(synthInputs("call(f, flag=yes)")).toEqual({ flag: "yes" });
});

test.each<[string, unknown]>([
  ["1e3", 1000],
  ["0x1F", 31n],
  ["0o17", 15n],
  ["+12", 12n],
  [".5", 0.5],
  ["1.", 1],
  [".inf", Infinity],
  ["~", null],
  ["2026-10-14", "2026-10-14"],
  ["2026-10-14T10:00:00Z", "2026-10-14T10:00:00Z"],
  ["None", "None"],
  ["none", "none"],
  ["NONE", "NONE"],
  ["tRUE", "tRUE"],
  ["-.nan", "-.nan"],
  ["yes", "yes"],
  ["30", 30n],
  ["null", null],
  ["true", true],
  ["hello world", "hello world"],
  // The named form reads a list here; a bare argument is one scalar.
  ["[1]", "[1]"],
])("bare keyword value %p reads with the flow YAML rules", (arg, expected) => {
  // `toBe` keeps a number and a bigint apart.
  expect(synthInputs(`call(f, x=${arg})`)["x"]).toBe(expected);
});

test("a bare nan keyword value can't be read", () => {
  const error = topicError("call(f, x=.nan)");
  expect(error.message).toBe(
    "use: call(f, ...) keyword 'x' value can't be read as YAML (NaN is not a yian value); quote it to keep it as text",
  );
  expect(error.cause).toBeInstanceOf(ArgumentValueError);
});

test("keyword value whitespace is trimmed", () => {
  expect(synthInputs("call(f, x= 1e3 )")["x"]).toBe(1000);
});

test.each([
  ['"1e3"', "1e3"],
  ["'~'", "~"],
  ['"None"', "None"],
  ["'2026-10-14'", "2026-10-14"],
])("quoted keyword value %p stays text", (arg, expected) => {
  expect(synthInputs(`call(f, x=${arg})`)["x"]).toBe(expected);
});

test("empty keyword value is null", () => {
  expect(synthInputs("call(f, x=)")).toEqual({ x: null });
});

test.each(["${a.output}", "hi ${a.output}"])("binding keyword value %p is unchanged", (arg) => {
  expect(synthInputs(`call(f, x=${arg})`)).toEqual({ x: arg });
});

test.each(["1e3", "0x1F", "~", "2026-10-14", "None", "tRUE", ".5"])(
  "named and inline forms read bare value %p alike",
  (value) => {
    const named = parseYamlLiteral(`x: ${value}`) as Record<string, unknown>;
    expect(synthInputs(`call(f, x=${value})`)["x"]).toBe(named["x"]);
  },
);

test("a 5000-digit bare int reads as a bigint in both forms", () => {
  const digits = "1".repeat(5000);
  expect(synthInputs(`call(f, x=${digits})`)["x"]).toBe(BigInt(digits));
  expect((parseYamlLiteral(`x: ${digits}`) as Record<string, unknown>)["x"]).toBe(BigInt(digits));
});

/** A flow with a `defs:` child `g` taking input `d`, and the given `nodes:` body. */
function gFlow(node: string, dType = "str"): string {
  return `id: f
name: f
input:
  cmd: !str
defs:
  g:
    input:
      d: !${dType}
    nodes:
      run:
        kind: code
        output: str
        code: |
          return "x"
    output: \${run.output}
nodes:
${node}`;
}

/** A `use` node binding `topic` to `binding`, and the flow output reading it. */
function bindingNode(binding: string): string {
  return (
    "  use:\n    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n" +
    `    input:\n      topic: ${binding}\n    output: str\n` +
    "output: ${use.output}\n"
  );
}

/** The `g` def built by hand: input `d` of type `dType`, returning a string. */
const gChild = (dType = "str") =>
  handChild({
    inputs: parametersFromYaml(`input:\n  d: !${dType}\n`, new Map()),
    producers: { run: null },
    output: "${run.output}",
  });

/** A `route` case node whose `then:` targets `target`, an `other` node, and the output. */
function caseNode(target: string): string {
  return (
    "  route:\n    kind: case\n    on: ${input.cmd}\n" +
    `    cases:\n      - when: a\n        then: ${target}\n` +
    "    else: other\n" +
    "  other:\n    kind: code\n    output: str\n    code: |\n      return 'other'\n" +
    "output: ${route.output}\n"
  );
}

test.each([
  ["binding", bindingNode, "use:", "topic: call("],
  ["case", caseNode, "case node 'route' then:", "then: call("],
])("an unreadable keyword value is a located load error: %s", (_, node, host, needle) => {
  const text = gFlow(node("call(g, d=2026-13-45)"));
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(
    `${host} call(g, ...) keyword 'd' value can't be read as YAML (not a valid date: month must be in 1..12); ` +
      "quote it to keep it as text",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(
    text.split("\n").findIndex((line) => line.includes(needle)) + 1,
  );
});

test("an out of range offset keyword value is a located load error", () => {
  const error = loadError(() => assembleText(gFlow(bindingNode("call(g, d=2026-01-01T10:00:00+05:99)"))));
  expect(error.message).toContain(
    "keyword 'd' value can't be read as YAML (not a valid datetime: UTC offset minutes must be under 60); " +
      "quote it to keep it as text",
  );
  expect(error.message).not.toContain("+06:39");
});

test.each(["str", "date"])("bare and quoted dates into call inputs of type %p load", (dType) => {
  assembleText(gFlow(bindingNode("call(g, d='2026-10-14')"), dType), "flow.yaml", { children: { g: gChild(dType) } });
  assembleText(gFlow(bindingNode("call(g, d=2026-10-14)"), dType), "flow.yaml", { children: { g: gChild(dType) } });
});

test("directive in outputs and asserts is desugared", () => {
  // A directive in the flow `output:` map and in a flow-level assert both desugar on the shared
  // traversal.
  const [descriptors, outputs, asserts] = desugarInlineCalls(
    new Map(),
    { result: "call(summarize, messages=${messages})" },
    { assertsSection: ["call(score, topic=${input.t})"], nextId: synthIdGen() },
  );
  expect(outputs).toEqual({ result: "${__call_0.output}" });
  expect(asserts[0]!.expression).toBe("${__call_1.output}");
  expect(asserts[0]!.display).toBe("call(score, topic=${input.t})");
  expect(synthIds(descriptors).map((callId) => synthOf(descriptors, callId).call)).toEqual(["summarize", "score"]);
});

test("directive rewrite preserves existing assertion presentation", () => {
  // Carrierized flow assertions retain presentation while executable text changes.
  const authored = new Assertion("call(score, topic=${input.t})", "score(topic) must pass", {
    origin: new OriginRef("asserts/0"),
  });

  const [, , assertions] = desugarInlineCalls(new Map(), null, { assertsSection: [authored], nextId: synthIdGen() });

  expect(assertions[0]!.expression).toBe("${__call_0.output}");
  expect(assertions[0]!.display).toBe(authored.display);
  expect(assertions[0]!.origin).toBe(authored.origin);
});

test("raw duplicate flow assertions become distinct ordered carriers", () => {
  // The front-end boundary retains occurrence identity even for equal YAML text.
  const text = "input.score > 0";

  const [, , assertions] = desugarInlineCalls(new Map(), null, { assertsSection: [text, text], nextId: synthIdGen() });

  expect(assertions).toEqual([new Assertion(text, text), new Assertion(text, text)]);
  expect(assertions[0]).not.toBe(assertions[1]);
});

// The hard break: a FLOW call embedded inside a `${...}` span (the retired `${flow(args)}` form)
// is a load error pointing at the call(...) directive. A pure builtin inside `${...}` stays legal.

test.each([
  ["whole value flow call in span", "${summarize(messages=messages)}", "summarize"],
  ["embedded flow call in comparison", "${score_one(t=input.t)} >= -1", "score_one"],
  ["embedded flow call in text", "pe=${relevance(t=input.t)}", "relevance"],
  ["coalesce of flow calls in span", "${a(x=1) | b(y=2)}", "a"],
])("%s is loud", (_name, value, callee) => {
  expect(topicError(value).message).toBe(
    `use: a flow call '${callee}' inside \${...} is not supported — write it as the whole value with the ` +
      `call(...) directive: call(${callee}, ...)`,
  );
});

test("flow call in outputs span is loud", () => {
  expect(() => desugarInlineCalls(new Map(), { result: "${summarize(m=m)}" }, { nextId: synthIdGen() })).toThrow(
    "flow outputs: a flow call 'summarize' inside ${...} is not supported",
  );
});

test.each([
  ["a keyword argument", "call(f, v=${g(1)})", "use: call(f) argument 'v'", "g"],
  ["a nested call's argument", "call(f, v=call(h, w=pre ${g(x=1)}))", "use: call(h) argument 'w'", "g"],
])("flow call in %s span is loud", (_name, value, where, callee) => {
  expect(topicError(value).message).toBe(
    `${where}: a flow call '${callee}' inside \${...} is not supported — write it as the whole value ` +
      `with the call(...) directive: call(${callee}, ...)`,
  );
});

test("pure builtin in span still allowed", () => {
  // `upper` is a builtin: a call to it inside `${...}` is NOT a flow call, so it passes through.
  const [descriptors] = desugarInlineCalls(nodes(code("use", { topic: "${upper(input.t)}" })), null, {
    nextId: synthIdGen(),
  });
  expect(descriptors.get("use")!.inputs["topic"]).toBe("${upper(input.t)}");
  expect(synthIds(descriptors)).toEqual([]);
});

/** A flow whose CODE node `use` echoes its `topic` binding. */
function echoFlow(binding: string, output = "${use.output}"): string {
  return `id: echo
name: echo
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: ${binding}
    output: str
output: ${output}
`;
}

test.each([
  ["'call(f, x=1'", "call(f, x=1"],
  ['"call(f) tail"', "call(f) tail"],
  ['"call(f, x=1)"', "call(f, x=1)"],
  ["|\n        call(f", "call(f\n"],
])("quoted binding %p runs as text", async (binding, text) => {
  const out = await runFlow(await loadConformanceFlow(echoFlow(binding)), {});
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(text);
});

test.each([
  ["call(f, x=1", "`call(` is never closed"],
  ["call(f) tail", "text follows the closing `)`: 'tail'"],
])("unquoted malformed binding %p is a located load error", (binding, problem) => {
  const error = loadError(() => assembleText(echoFlow(binding)));
  expect(error.message).toContain(`use: ${problem}`);
  expect(error.message).toContain('If you meant the text, quote the whole value: `"call(f');
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(8);
});

test("quoted flow output runs as text", async () => {
  const out = await runFlow(await loadConformanceFlow(echoFlow("hi", "'call(f'")), {});
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe("call(f");
});

test("unquoted malformed flow output is loud", () => {
  expect(() => assembleText(echoFlow("hi", "call(f) tail"))).toThrow("flow outputs: text follows the closing");
});

test("unquoted malformed map over is loud", () => {
  const text = `id: m
name: m
nodes:
  many:
    kind: map
    call: f
    over: call(g
    input:
      topic: \${item}
output: \${many.output}
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toContain("many: `call(` is never closed");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(7);
});

test("unquoted directive holding a separator is a call, not a typed binding", () => {
  const text = `id: spaced
name: spaced
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(f, topic = "a = b")
    output: str
output: \${use.output}
`;
  const f = handChild({
    inputs: parametersFromYaml("input:\n  topic: !str\n", new Map()),
    producers: { e: null },
    output: "${e.output}",
  });
  const { flow } = assembleText(text, "flow.yaml", { children: { f } });
  expect(flow.compiled.wiring.get("__call_0")!.get("topic")!.source).toBe("a = b");
});

test.each([
  ["call(f, t=${input.x} == X)", "t", "${input.x} == X"],
  ["call(f, t=a>=b, u=a<=b, v=a!=b)", "t", "a>=b"],
  ["call(f, t=a=b)", "t", "a=b"],
  ["call(f, note=a = b)", "note", "a = b"],
])("comparison and glued equals in keyword value %p stay text", (value, key, expected) => {
  expect(synthInputs(value)[key]).toBe(expected);
});

test.each([
  ["call(f, a=${x} b=${y})", "keyword 'a' value '${x} b=${y}' looks like two keywords"],
  ["call(f, =1)", "argument '=1' needs one name before `=`"],
  ["call(f, a b=1)", "argument 'a b=1' needs one name before `=`"],
])("missing comma or name in keywords %p is loud", (value, problem) => {
  expect(topicError(value).message).toContain(problem);
});

test("no quote hint when quoting would not give text", () => {
  // Quoting a span-bearing call is its own error.
  expect(topicError("call(f, topic=${input.x}").message).not.toContain("If you meant the text");
});

test("typed text holding a call is the old form error", () => {
  expect(loadError(() => assembleText(echoFlow("str = call(f"))).message).toContain(
    "binding 'topic': write the type as a tag: `topic: !str call(f`",
  );
  expect(loadError(() => assembleText(echoFlow("hi", "str = call(f"))).message).toContain(
    "write the type as a tag: `output: !str call(f`",
  );
});

// ---------- harvested conformance cases ----------

test.each([
  "load-types/compose_tagged_bindings__malformed_unquoted_call_after_a_tag",
  "load-map-call/engine_call_directive__an_unreadable_date_keyword_value",
  "load-map-call/engine_call_directive__private_callee_in_a_binding_is_rejected",
  "load-map-call/engine_call_directive__text_after_a_binding_call",
  "load-map-call/engine_call_directive__unquoted_malformed_map_over_is_rejected",
])("assembled: %s", (id) => {
  runCase({ id, call: assembled });
});
