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
 * `load-map-call/seed_21_inline_call`, pinned at the end. `test_inline_call_e06_type_mismatch_is_loud`
 * pins the whole message, where Python matches fragments; its harvested case
 * `load-map-call/engine_inline_calls__inline_call_argument_source_type_mismatch_is_rejected` is
 * pinned at the end too.
 *
 * Ported through `loadFlow` (the fixture CODE references resolve against the conformance dir):
 * the load halves of `test_inline_call_resolves_through_external_resolver` (its resolver also
 * loads the external flow) and `test_inline_call_inside_defs_body_runs` (the def's synth call is
 * read off the baked child), and `test_inline_call_unknown_callee_is_loud`, which pins the whole
 * message where Python matches fragments.
 * Later (T4): the run half of every load-and-run row.
 *
 * The origin rows locate each synth call node and its argument bindings at the authored binding
 * the `call(...)` was written in: the load halves of
 * `test_inline_call_rewrite_keeps_host_site_and_generated_call_shares_it`
 * (`tests/compose/test_node_input_origins.py`) and
 * `test_boundary_nodes_are_originless_and_an_inline_call_shares_its_binding`
 * (`tests/compose/test_node_origins.py`), neither file having a TS mirror yet, and TS-only rows for
 * nested calls in a node input, a flow output, a flow assert and the `then:`/`else:` targets of a
 * CASE. Origins resolve through the assembly's catalog builder.
 *
 * TS-only: the harvested cases the inline-call desugar raises whose Python test file has no TS
 * mirror, each named by its case id (from `test_expression_nesting.py`, `test_limit_overrides.py`,
 * `test_constructor_load.py`, `test_any_name_literals.py` and this file), pinned at the end.
 */

import { afterEach, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { desugarInlineCalls } from "../../src/compose/desugar/inline_calls.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  handChild,
  loadConformanceFlow,
  loadError,
  loadErrorAsync,
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

/** Each call node's id mapped to its callee flow id. */
const callees = (compiled: ReturnType<typeof load>) =>
  Object.fromEntries(
    [...compiled.nodes].flatMap(([nodeId, node]) => (node instanceof CallNode ? [[nodeId, node.flowId]] : [])),
  );

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
  expect(callees(compiled)).toEqual({ __call_0: "enrich" }); // the inline call became a synth node
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
  expect(callees(compiled)).toEqual({ __call_0: "enrich" });
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
  expect(callees(compiled)).toEqual({ __call_0: "seed" });
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
  expect(callees(compiled)).toEqual({ __call_0: "inner", __call_1: "outer" }); // inner minted first
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
  expect(callees(compiled)).toEqual({ __call_0: "enrich" });
});

test("inline call resolves through external resolver", async () => {
  const external = `
id: ext_flow
name: ext_flow
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
  const text = `
id: ic_ext
name: ic_ext
input:
  topic: !str
uses:
  ext_flow: ext_flow
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(ext_flow, topic=\${input.topic})
    output: str
output: \${use.output}
`;
  const asked: string[] = [];
  const loaded = await loadConformanceFlow(text, {
    childResolver: (flowId) => {
      asked.push(flowId);
      return loadConformanceFlow(external);
    },
  });
  expect(asked).toEqual(["ext_flow"]);
  expect(callees(loaded.compiled)).toEqual({ __call_0: "ext_flow" });
});

test("inline call inside defs body loads", async () => {
  // The desugar runs for def bodies too (shared assembly); the def's synth ids live in their own
  // namespace. Here `enrich`'s own node hosts an inline call to `helper`.
  const text = `
id: ic_defbody
name: ic_defbody
input:
  topic: !str
defs:
  helper:
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
  enrich:
    input:
      topic: !str
    nodes:
      y:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: call(helper, topic=\${input.topic})
        output: str
    output: \${y.output}
nodes:
  go:
    kind: call
    call: enrich
    input:
      topic: \${input.topic}
output: \${go.output}
`;
  const { compiled } = await loadConformanceFlow(text);
  expect(callees(compiled)).toEqual({ go: "enrich" });
  const enrich = (compiled.nodes.get("go") as CallNode).child as ReturnType<typeof load>;
  expect(callees(enrich)).toEqual({ __call_0: "helper" });
});

test("inline call unknown callee is loud", async () => {
  const text = `
id: ic_unk
name: ic_unk
input:
  topic: !str
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(nope, x=\${input.topic})
    output: str
output: \${use.output}
`;
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toBe(
    "callable 'nope' is neither an in-file defs: callable nor a uses: alias — external flows are reachable " +
      "only through a uses: alias (declare `uses: {nope: <ref>}`)",
  );
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
  expect(loadError(() => assembleText(text)).message).toBe(
    "node 'each': inline call 'wrap' cannot capture ${item} (arg 't') — it desugars to a top-level call node " +
      "with no map-element scope; use a named `call` node (whose per-element `input:` may read `${item}`)",
  );
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
  expect(error.message).toBe(
    "use: call(enrich, ...) passes a positional argument '30'; every argument after the flow id must be a " +
      'keyword (name=value). If you meant the text, quote the whole value: `"call(enrich, 30)"`',
  );
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
  expect(error.message).toBe(
    "flow outputs: call(enrich, ...) passes a positional argument '30'; every argument after the flow id must " +
      'be a keyword (name=value). If you meant the text, quote the whole value: `"call(enrich, 30)"`',
  );
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
  expect(callees(compiled)).toEqual({ __call_0: "enrich", __call_1: "enrich" }); // distinct, no collision
});

test("inline call e06 type mismatch is loud", () => {
  // A synth call binding goes through the same cross-flow type check as a named call, which names
  // the authored argument, not the synth node.
  const error = loadError(() =>
    load(`
id: ic_e06
name: ic_e06
input:
  topic: !str
${ENRICH_DEF}
nodes:
  num:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.topic}
    output: int
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic=\${num.output})
    output: str
output: \${use.output}
`),
  );

  expect(error.message).toBe(
    "node 'use' input 'topic' call(enrich) argument 'topic' — child expects 'str', source is 'int'",
  );
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

test("assembled: load-map-call/engine_inline_calls__inline_call_argument_source_type_mismatch_is_rejected", () => {
  runCase({
    id: "load-map-call/engine_inline_calls__inline_call_argument_source_type_mismatch_is_rejected",
    call: assembledWith(ENRICH),
  });
});

// ---------- origins: a synth call shares the authored binding it was written in ----------

/** Return node `nodeId`'s wiring binding `name` of `compiled` as the `Binding` it is. */
function bindingOf(compiled: ReturnType<typeof load>, nodeId: string, name: string): Binding {
  const binding = compiled.wiring.get(nodeId)!.get(name);
  expect(binding).toBeInstanceOf(Binding);
  return binding as Binding;
}

// From tests/compose/test_node_input_origins.py.
test("inline call rewrite keeps host site and generated call shares it", () => {
  const text = `id: rewritten-input-origin
name: rewritten-input-origin
defs:
  helper:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: !int
    output: int
nodes:
  consumer:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: call(helper, x=3)
    output: int
output: \${consumer.output}
`;
  const hostCursor = SourceDocument.parse(text).root.child("nodes").child("consumer").child("input").child("x");
  const { flow, builder } = assembleText(text, "flow.yaml", { children: { helper: def("  x: !int\n", "int") } });
  const compiled = flow.compiled;

  const host = compiled.nodes.get("consumer")!;
  const hostBinding = bindingOf(compiled, "consumer", "x");
  const parameter = host.params!.find((param) => param.name === "x")!;
  expect(parameter.origin).toBeInstanceOf(OriginRef);
  expect(hostBinding.origin).toBeInstanceOf(OriginRef);
  expect(hostBinding.source).toBe("${__call_0.output}");
  expect(builder.resolveRegistered(parameter.origin!)!.range).toEqual(hostCursor.keyRange!);
  expect(builder.resolveRegistered(hostBinding.origin!)!.range).toEqual(hostCursor.range!);
  expect(builder.resolveRegistered(hostBinding.origin!)!.text).toBe(text);

  // The generated call and its argument sit inside the host's binding, so they share it.
  const generated = compiled.nodes.get("__call_0")!;
  const generatedBinding = bindingOf(compiled, "__call_0", "x");
  expect(generated.params!.find((param) => param.name === "x")!.origin).toBeNull();
  expect(generatedBinding.source).toBe(3n);
  expect(generatedBinding.origin).toBe(hostBinding.origin);
  expect(generated.origin).toBe(hostBinding.origin);
});

// From tests/compose/test_node_origins.py.
test("boundary nodes are originless and an inline call shares its binding", () => {
  const text = `id: generated-origins
name: generated-origins
input:
  topic: !str
defs:
  echo:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: !str
    output: str
nodes:
  consumer:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(echo, topic=\${input.topic})
    output: str
output: \${consumer.output}
`;
  const compiled = load(text, { echo: def("  topic: !str\n", "str") });
  const nodes = compiled.nodes;

  expect(nodes.get(START_ID)!.origin).toBeNull();
  expect(nodes.get(END_ID)!.origin).toBeNull();
  expect(nodes.get("consumer")!.origin).toBeInstanceOf(OriginRef);
  const hostBindingOrigin = bindingOf(compiled, "consumer", "topic").origin;
  expect(hostBindingOrigin).toBeInstanceOf(OriginRef);
  expect(nodes.get("__call_0")!.origin).toBe(hostBindingOrigin);
});

test("nested calls in a node input take the binding's origin", () => {
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
  const origin = bindingOf(compiled, "use", "topic").origin;
  expect(origin).toBeInstanceOf(OriginRef);
  for (const callId of ["__call_0", "__call_1"]) {
    expect(compiled.nodes.get(callId)!.origin).toBe(origin);
    expect(bindingOf(compiled, callId, "topic").origin).toBe(origin);
  }
});

test("a flow output call takes the output binding's origin", () => {
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
  const origin = compiled.outputs[0]!.binding.origin;
  expect(origin).toBeInstanceOf(OriginRef);
  expect(compiled.nodes.get("__call_0")!.origin).toBe(origin);
  expect(bindingOf(compiled, "__call_0", "topic").origin).toBe(origin);
});

test("a flow assert call takes the assertion's origin", () => {
  const text = `
id: ic_assert
name: ic_assert
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
output: \${seed.output}
asserts:
  - call(enrich, topic=\${input.topic})
`;
  const item = SourceDocument.parse(text).root.child("asserts").child(0);
  const { flow, builder } = assembleText(text, "flow.yaml", { children: ENRICH });
  const compiled = flow.compiled;

  const origin = compiled.nodes.get("__call_0")!.origin;
  expect(origin).toBeInstanceOf(OriginRef);
  expect(builder.resolveRegistered(origin!)!.range).toEqual(item.range!);
  expect(bindingOf(compiled, "__call_0", "topic").origin).toBe(origin);
});

test("a then: call registers its branch site once and a nested call reuses it", () => {
  const topic = def("  topic: !str\n", "str");
  const text = `
id: ic_branch
name: ic_branch
input:
  go: !bool
  topic: !str
nodes:
  gate:
    kind: case
    cases:
      - when: \${input.go}
        then: call(outer, topic=call(inner, topic=\${input.topic}))
    else: call(take, topic=\${input.topic})
output: \${gate.output}
`;
  const gate = SourceDocument.parse(text).root.child("nodes").child("gate");
  const { flow, builder } = assembleText(text, "flow.yaml", { children: { outer: topic, inner: topic, take: topic } });
  const compiled = flow.compiled;

  // __call_0 is the inner call, __call_1 the outer one (inner first), __call_2 the else call.
  const thenOrigin = compiled.nodes.get("__call_1")!.origin;
  expect(thenOrigin).toBeInstanceOf(OriginRef);
  expect(builder.resolveRegistered(thenOrigin!)!.range).toEqual(gate.child("cases").child(0).child("then").range!);
  for (const callId of ["__call_0", "__call_1"]) {
    expect(compiled.nodes.get(callId)!.origin).toBe(thenOrigin);
    expect(bindingOf(compiled, callId, "topic").origin).toBe(thenOrigin);
  }
  const elseOrigin = compiled.nodes.get("__call_2")!.origin;
  expect(builder.resolveRegistered(elseOrigin!)!.range).toEqual(gate.child("else").range!);
  expect(bindingOf(compiled, "__call_2", "topic").origin).toBe(elseOrigin);
});
