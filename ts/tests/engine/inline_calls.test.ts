/**
 * Inline `call(...)` directive calls, end to end: a port of `tests/engine/test_inline_calls.py`.
 *
 * An inline call is a whole-value `call(<flow-id>, kw=...)` directive that desugars at load into
 * an anonymous `call` node, with the host binding rewritten to `${<synth>.output}`. The load
 * halves assemble the flow (`assembleText`), each `defs:` callee built by hand (`handChild`) with
 * the signature the row reads; the direct row drives `desugarInlineCalls`.
 *
 * Changed: Python loads and runs each flow; these rows check the load: the synth node, its edges,
 * and the argument binding a run would read. `test_inline_call_seed_21_loads` is the harvested
 * `load-map-call/seed_21_inline_call`, pinned at the end.
 *
 * Later (T3-22): `test_inline_call_resolves_through_external_resolver` (`uses:` and a child
 * resolver), `test_inline_call_inside_defs_body_runs` (a `defs:` body) and
 * `test_inline_call_unknown_callee_is_loud` (`defs:` resolution of an unknown callee).
 * Later (T3-19): `test_inline_call_e06_type_mismatch_is_loud` (the typed-binding check).
 * Later (T4): the run half of every load-and-run row.
 *
 * TS-only: the harvested cases the inline-call desugar raises whose Python test file has no TS
 * mirror, each named by its case id (from `test_expression_nesting.py`, `test_limit_overrides.py`,
 * `test_constructor_load.py`, `test_any_name_literals.py` and this file), pinned at the end.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { desugarInlineCalls } from "../../src/compose/desugar/inline_calls.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { forget } from "../../src/defaults.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

afterEach(() => forget());

/**
 * A def built by hand: `input:` text `inputs`, one producer `x` of type `output` (a type text read
 * against `registry`), returned.
 */
function def(inputs: string, output: string, registry: TypeRegistry = new Map()): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml(`input:\n${inputs}`, new Map()),
    producers: { x: new ValueContract(typeFor(output, registry)) },
    output: "${x.output}",
  });
}

// An in-file `enrich` def that echoes its `topic`: the callee for the inline calls.
const ENRICH_DEF = `
defs:
  enrich:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.topic}
        output: str
    output: \${x.output}
`;
const ENRICH = { enrich: def("  topic: !str\n", "str") };

/** Assemble `text` with the `enrich` def; return the compiled flow. */
const load = (text: string, children: Record<string, ComposedFlow> = ENRICH) =>
  assembleText(text, "flow.yaml", { children }).flow.compiled;

/** The `[from, to]` pairs of a compiled flow's edges. */
const edgePairs = (compiled: ReturnType<typeof load>) => compiled.edges.map((edge) => [edge.from, edge.to]);

test("inline call in node input loads", () => {
  const compiled = load(`
id: ic1
name: ic1
input:
  topic: !str
${ENRICH_DEF}
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic=\${input.topic})
    output: str
output: \${use.output}
`);
  expect(compiled.nodes.has("__call_0")).toBe(true); // the inline call became a synth node
});

test("inline call in human_input synthesizes and wires the call", () => {
  const compiled = load(`
id: ic_human_input
name: ic_human_input
input:
  topic: !str
${ENRICH_DEF}
nodes:
  ask:
    kind: human_input
    input:
      context: call(enrich, topic=\${input.topic})
    prompt: "Review \${context}"
    output: str
output: \${ask.output}
`);
  expect(compiled.nodes.has("__call_0")).toBe(true);
  expect(edgePairs(compiled)).toContainEqual(["__call_0", "ask"]);
});

test("inline call in loop input synthesizes and wires the call", () => {
  const compiled = load(
    `
id: ic_loop
name: ic_loop
input:
  start: !int
defs:
  seed:
    input:
      n: !int
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.n}
        output: int
    output: \${echo.output}
  body:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_countdown
        input:
          n: \${input.n}
        output:
          n: int
    output: \${step.output}
nodes:
  repeat:
    kind: loop
    call: body
    input:
      n: call(seed, n=\${input.start})
    times: 1
output: \${repeat.output}
`,
    { seed: def("  n: !int\n", "int"), body: def("  n: !int\n", "Step", readTypedefs({ Step: { n: "int" } })) },
  );
  expect(compiled.nodes.has("__call_0")).toBe(true);
  expect(edgePairs(compiled)).toContainEqual(["__call_0", "repeat"]);
});

test("inline call literal arg binds its text", () => {
  // A bare literal arg (string) flows like the named form's `input: {topic: HARD}`.
  const compiled = load(`
id: ic_lit
name: ic_lit
input:
  topic: !str
${ENRICH_DEF}
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic="HARDCODED")
    output: str
output: \${use.output}
`);
  expect(compiled.wiring.get("__call_0")!.get("topic")!.source).toBe("HARDCODED");
});

test("inline call fan-in infers edges", () => {
  const compiled = load(`
id: ic_fanin
name: ic_fanin
input:
  topic: !str
${ENRICH_DEF}
nodes:
  y:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.topic}
    output: str
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic=\${y.output})
    output: str
output: \${use.output}
`);
  expect(edgePairs(compiled)).toContainEqual(["y", "__call_0"]); // y -> synth call (the fan-in)
  expect(edgePairs(compiled)).toContainEqual(["__call_0", "use"]); // synth call -> consumer
});

test("nested inline calls load", () => {
  const topic = def("  topic: !str\n", "str");
  const compiled = load(
    `
id: ic_nested
name: ic_nested
input:
  topic: !str
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(outer, topic=call(inner, topic=\${input.topic}))
    output: str
output: \${use.output}
`,
    { inner: topic, outer: topic },
  );
  expect(compiled.nodes.has("__call_0")).toBe(true); // inner (minted first)
  expect(compiled.nodes.has("__call_1")).toBe(true); // outer
});

test("inline call in flow output loads", () => {
  const compiled = load(`
id: ic_out
name: ic_out
input:
  topic: !str
${ENRICH_DEF}
nodes:
  seed:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.topic}
    output: str
output: call(enrich, topic=\${seed.output})
`);
  expect(compiled.nodes.has("__call_0")).toBe(true);
});

// ---------- guards: `${item}` capture, the reserved synth-id prefix, a malformed call ----------

test("inline call capturing item is loud", () => {
  // An inline call lifts to a top-level synth node with no map-element scope, so an arg reading
  // ${item} is rejected: use a named `call` node instead.
  const text = `
id: ic_item
name: ic_item
input:
  topics: !list(str)
nodes:
  each:
    kind: map
    call: one
    over: \${input.topics}
    input:
      topic: call(wrap, t=\${item})
output: \${each.output}
`;
  const message = loadError(() => assembleText(text)).message;
  expect(message).toContain("item");
  expect(message).toContain("named `call`");
  expect(message).toContain("`input:` may read");
});

test("user node with the synth prefix is loud", () => {
  const text = `
id: ic_collide
name: ic_collide
input:
  topic: !str
nodes:
  __call_0:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.topic}
    output: str
output: \${__call_0.output}
`;
  expect(loadError(() => assembleText(text)).message).toBe(
    "node id '__call_0' uses the reserved inline-call prefix '__call_'",
  );
});

test("inline call positional arg is loud and located", () => {
  const text = `
id: ic_pos
name: ic_pos
input:
  topic: !str
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, 30)
    output: str
output: \${use.output}
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toContain("keyword");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(11);
});

// ---------- named-form parity, located flow-output errors, binding positions ----------

test("quoted interpolated arg strips quotes like the named form", () => {
  // A quoted arg with interpolation `"hi ${name}"` unwraps to the template `hi ${name}` (quotes
  // stripped), matching the named form's YAML quoted scalar: no literal quotes.
  const compiled = load(`
id: ic_qi
name: ic_qi
input:
  name: !str
${ENRICH_DEF}
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic="hi \${input.name}")
    output: str
output: \${use.output}
`);
  expect(compiled.wiring.get("__call_0")!.get("topic")!.source).toBe("hi ${input.name}");
});

test("malformed inline call in outputs is located", () => {
  const text = `
id: ic_outloc
name: ic_outloc
input:
  topic: !str
nodes:
  seed:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.topic}
    output: str
output: call(enrich, 30)
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toContain("keyword");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(13);
});

test("inline call in tool inputs desugars", () => {
  // The TOOL descriptor's `inputs` binding field desugars too.
  const descriptors = new Map([
    ["t", new ToolDescriptor({ id: "t", toolId: "some_tool", inputs: { q: "call(enrich, x=${input.k})" } })],
  ]);
  const [newDescriptors] = desugarInlineCalls(descriptors, "${t.output}");
  expect(newDescriptors.get("t")!.inputs["q"]).toBe("${__call_0.output}");
  const synth = newDescriptors.get("__call_0") as CallDescriptor;
  expect(synth).toBeInstanceOf(CallDescriptor);
  expect(synth.call).toBe("enrich");
  expect(synth.inputs).toEqual({ x: "${input.k}" });
  expect(synth.over).toBeNull();
});

test("inline call in mapped call over wires the producer into the map", () => {
  const compiled = load(
    `
id: ic_over
name: ic_over
input:
  topics: !list(str)
nodes:
  each:
    kind: map
    call: one
    over: call(producelist, xs=\${input.topics})
    input:
      topic: \${item}
output: \${each.output}
`,
    { one: def("  topic: !str\n", "str"), producelist: def("  xs: !list(str)\n", "list(str)") },
  );
  expect(edgePairs(compiled)).toContainEqual(["__call_0", "each"]); // the synth-over producer -> the map
});

test("two host sites mint unique ids", () => {
  const compiled = load(`
id: ic_two
name: ic_two
input:
  topic: !str
${ENRICH_DEF}
nodes:
  a:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic=\${input.topic})
    output: str
  b:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic=\${input.topic})
    output: str
output:
  ra: \${a.output}
  rb: \${b.output}
`);
  expect(compiled.nodes.has("__call_0")).toBe(true); // distinct, no collision
  expect(compiled.nodes.has("__call_1")).toBe(true);
});

// ---------- harvested conformance cases ----------

// The inline-call desugar raises each of these before any callee resolves.
test.each<CaseRow>(
  [
    "load-core/compose_expression_nesting__too_deep_binding",
    "load-core/compose_expression_nesting__too_deep_flow_output",
    "load-core/engine_limit_overrides__expression_nesting_follows_the_value_limit",
    "load-types/compose_constructor_load__a_builtin_type_name_in_a_binding_is_not_a_conversion",
    "load-types/compose_constructor_load__a_flow_call_keeps_the_flow_call_message",
    "load-types/compose_constructor_load__a_span_inside_a_constructor_shows_the_rewrite",
    "load-map-call/compose_any_name_literals__a_malformed_call_with_a_credential_keyword",
    "load-map-call/engine_inline_calls__inline_call_capturing_item_is_rejected",
    "load-map-call/engine_inline_calls__positional_call_arg_in_a_node_input_is_rejected",
    "load-map-call/engine_inline_calls__positional_call_arg_in_the_flow_output_is_rejected",
    "load-map-call/engine_inline_calls__user_node_with_synth_prefix_is_rejected",
  ].map((id) => ({ id, call: assembled })),
)("assembled: $id", runCase);

test("assembled: load-map-call/seed_21_inline_call", () => {
  // `enrich` returns its `fetch` node's headlines, a `list(str)`.
  runCase({
    id: "load-map-call/seed_21_inline_call",
    call: assembledWith({ enrich: def("  topic: !str\n", "list(str)") }),
  });
});
