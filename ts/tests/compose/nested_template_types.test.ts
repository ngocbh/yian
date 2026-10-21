/**
 * Types of nested binding values are inferred from their structure: a port of
 * `tests/compose/test_nested_template_types.py`.
 *
 * A mapping or list binding holding `${...}` spans is a `ValueTemplate`. Its type is inferred field
 * by field: a nested typed binding gives its declared type, a string its template type, a literal
 * its kind (its exact value only where a tag checks it). A tag on the whole value, a CALL child's
 * parameter, a TOOL's stamped input types, a LOOP's carried record and an inline body's generated
 * parameters all read that structural type.
 *
 * Ported: every row but those below, the parametrized ones with all their rows. The CODE bodies
 * are Python's, which read the same in JS (decision 11), but for `return null`; a CODE body gets
 * an int as a `number` (decision 45) and a whole-number return reads back as an int, so the outputs
 * hold `bigint`s (decision 4a). `test_nested_typed_binding_is_checked_at_its_own_line` is ported
 * here; `typed_bindings_check.test.ts` has a flow-style variant of it.
 *
 * Covered by `load-types/compose_nested_template_types__tagged_template_with_a_disjoint_field_fails`:
 * `test_tagged_template_with_a_disjoint_field_fails`; by
 * `load-types/compose_nested_template_types__template_into_a_record_child_param_loads`:
 * `test_template_into_a_record_child_param_loads`; by
 * `load-types/compose_nested_template_types__an_empty_list_in_an_inline_call_template` (the quoted
 * key in the text form, decision 43): `test_inline_call_template_arg_with_no_proven_type_suggests_a_tag`;
 * by `load-types/compose_nested_template_types__nested_item_read_is_typed_from_the_map_element`:
 * `test_nested_item_read_is_typed_from_the_map_element`; by
 * `load-types/compose_nested_template_types__inline_call_template_list_of_mixed_types_fails`:
 * `test_inline_call_template_list_of_mixed_types_fails`; by
 * `load-types/compose_nested_template_types__wrong_literal_value_under_a_tag_fails_at_load`:
 * `test_wrong_literal_value_under_a_tag_fails_at_load`; by
 * `run-core/compose_nested_template_types__partial_template_into_a_child_param_fails` (its flow
 * keeps only the typedef it reads): `test_partial_template_into_a_child_param_is_checked_at_runtime`.
 */

import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import type { ValueContract } from "../../src/nodes/contracts.ts";
import type { ToolNode } from "../../src/nodes/tool/node.ts";
import { runFlow } from "../../src/run.ts";
import { canonicalTypeLabel, type Type } from "../../src/typesys/values.ts";
import { loadConformanceFlow, loadErrorAsync } from "./declarations.ts";

const TYPEDEFS = `typedefs:
  Rec:
    a: str
    n: int
  Buy:
    conviction: float
  Obj:
    n: int
  Trade:
    action: Literal(buy)
    conviction: float
`;

/** Return a flow whose CODE node `c` binds `x` as `binding`, reading `p`. */
function codeFlow(binding: string, options: { pOutput?: string; extra?: string } = {}): string {
  const { pOutput = "str", extra = "" } = options;
  return (
    TYPEDEFS +
    `id: nested_types
name: nested_types
nodes:
  p:
    kind: code
    output: ${pOutput}
    code: |
      return "hi"
  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs["x"]
${extra}output: \${c.output}
`
  );
}

/** Load `text`, expecting a load error, and return it. */
const loadErrorOf = (text: string): Promise<LoadError> => loadErrorAsync(loadConformanceFlow(text));

/** Return the 1-based start line of a load error's innermost frame. */
const errorLine = (error: LoadError): number => error.diagnostic.frames.at(-1)!.range.startLine;

/** Return the 1-based number of the first line of `text` equal to `line`. */
const lineOf = (text: string, line: string): number => text.split("\n").indexOf(line) + 1;

/** Return each entry's type as its canonical label. */
const labels = (types: ReadonlyMap<string, Type>): Record<string, string> =>
  Object.fromEntries([...types].map(([name, type]) => [name, canonicalTypeLabel(type)]));

/** Return the canonical label of the one generated parameter of inline CALL node `c`. */
function onlyParamLabel(loaded: Awaited<ReturnType<typeof loadConformanceFlow>>): string {
  const params = (loaded.compiled.nodes.get("c") as CallNode).childInputs;
  expect(params).toHaveLength(1);
  return canonicalTypeLabel(params[0]!.contract!.type);
}

// ---------- a tag on the whole template ----------

test("tagged template with a reference field loads", async () => {
  await loadConformanceFlow(codeFlow("!Rec {a: '${p.output}', n: 1}"));
});

test("tagged template missing a required field names it", async () => {
  const error = await loadErrorOf(codeFlow("!Rec {a: '${p.output}'}"));

  expect(error.diagnostic.message).toBe(
    "node 'c' input 'x': declared Rec, but the value is " + '{"a":str} (missing required `n`)',
  );
});

test("tagged template with a str reference into a float fails", async () => {
  const error = await loadErrorOf(codeFlow("!Buy {conviction: '${p.output}'}"));

  expect(error.diagnostic.message).toBe("node 'c' input 'x': declared Buy, but the value is " + '{"conviction":str}');
});

test("tagged template with an optional reference fails at runtime", async () => {
  const flow = await loadConformanceFlow(
    codeFlow("!Buy {conviction: '${p.output}'}", { pOutput: "Optional(float)" }).replace('return "hi"', "return null"),
  );

  const result = await runFlow(flow, {});

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
});

test("nested typed binding is checked at its own line", async () => {
  const text = codeFlow("\n        a: !int '${p.output}'\n        n: 1");

  const error = await loadErrorOf(text);

  expect(error.diagnostic.message).toBe("node 'c' input 'x.a': declared int, but the value is str");
  expect(errorLine(error)).toBe(lineOf(text, "        a: !int '${p.output}'"));
});

test("nested typed literal fits the outer tag", async () => {
  await loadConformanceFlow(codeFlow("!Rec {a: '${p.output}', n: !int 3}"));
});

// ---------- CALL children, TOOL stamps, and child outputs ----------

/** Return a flow whose CALL node `c` binds the child's `p` as `binding`. */
function callFlow(binding: string, param: string, options: { nOutput?: string } = {}): string {
  const { nOutput = "str" } = options;
  return (
    TYPEDEFS +
    `id: nested_call
name: nested_call
defs:
  child:
    input:
      p: ${param}
    nodes: {}
    output: \${input.p}
nodes:
  n:
    kind: code
    output: ${nOutput}
    code: |
      return "hi"
  c:
    kind: call
    call: child
    input:
      p: ${binding}
output: \${c.output}
`
  );
}

test("template with a disjoint field into a child param fails", async () => {
  const error = await loadErrorOf(callFlow("{a: '${n.output}', n: 1}", "!Rec", { nOutput: "int" }));

  expect(error.message).toContain("call node 'c': binding 'p' — child expects");
});

test("tagged empty list into a list child param loads", async () => {
  const text = callFlow("{items: !list(str) []}", "!Items").replace(
    "typedefs:\n",
    "typedefs:\n  Items:\n    items: list(str)\n",
  );

  await loadConformanceFlow(text);
});

/** Return a flow whose TOOL node `t` binds `v` as `binding`. */
const toolFlow = (binding: string) => `id: stamped
name: stamped
input:
  n: !int
  a: !str
  o: !Optional(str)
nodes:
  t:
    kind: tool
    tool_id: place_order
    input:
      v: ${binding}
output: \${t.output}
`;

test.each([
  ["{a: '${input.a}', n: '${input.n}'}", '{"a":str,"n":int}'],
  ["['${input.a}', 'x ${input.n}']", "list(str)"],
  ["['${input.n}', 2]", "list(int)"],
  ["['${input.o}', '${input.a}']", "list(Optional(str))"],
  ["{a: ['${input.a}'], b: '${input.n}'}", '{"a":list(str),"b":int}'],
])("tool template input is stamped with its structural type: %s", async (binding, label) => {
  const node = (await loadConformanceFlow(toolFlow(binding))).compiled.nodes.get("t") as ToolNode;

  expect(labels(node.inputTypes)).toEqual({ v: label });
});

test.each(["['${input.a}', '${input.n}']", "{a: '${input.a}', b: []}"])(
  "tool template input with no exact type is absent: %s",
  async (binding) => {
    const node = (await loadConformanceFlow(toolFlow(binding))).compiled.nodes.get("t") as ToolNode;

    expect(node.inputTypes.size).toBe(0);
  },
);

const CHILD_OUTPUT = `id: nested_child_output
name: nested_child_output
defs:
  child:
    nodes:
      n:
        kind: code
        output: str
        code: |
          return "hi"
    output:
      a:
        b: \${n.output}
nodes:
  sub:
    kind: call
    call: child
  c:
    kind: code
    input:
      x: READ
    code: |
      return inputs["x"]
output: \${c.output}
`;

test("child template output types the parent read", async () => {
  const error = await loadErrorOf(CHILD_OUTPUT.replace("READ", "!int ${sub.output.a.b}"));

  expect(error.diagnostic.message).toBe("node 'c' input 'x': declared int, but the value is str");
});

test("child template output rejects an unknown field read", async () => {
  const error = await loadErrorOf(CHILD_OUTPUT.replace("READ", "${sub.output.a.zz}"));

  expect(error.diagnostic.message).toBe(
    "flow has unresolved references:\n" +
      "  node 'c' input 'x' from: reference ${sub.output.a.zz} reads unknown field 'zz' on record",
  );
});

const SKIPPED_CHILD_OUTPUT = `id: nested_skipped_child
name: nested_skipped_child
defs:
  child:
    input:
      go: !bool
    nodes:
      gate:
        kind: case
        cases:
          - when: \${input.go}
            then: n
      n:
        kind: code
        output: str
        code: |
          return "hi"
    output:
      a:
        b: \${n.output}
nodes:
  sub:
    kind: call
    call: child
    input:
      go: true
  c:
    kind: code
    input:
      x: !str \${sub.output.a.b}
    code: |
      return inputs["x"]
output: \${c.output}
`;

test("a skippable nested end read is optional", async () => {
  const loaded = await loadConformanceFlow(SKIPPED_CHILD_OUTPUT);

  const contract = loaded.compiled.nodes.get("sub")!.outputContract as ValueContract;
  expect(canonicalTypeLabel(contract.type)).toBe('{"a":{"b":Optional(str)}}');
});

// ---------- LOOP seeds ----------

const LOOP = `id: nested_loop
name: nested_loop
input:
  start: !int
defs:
  step:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: \${bump.output}
nodes:
  repeat:
    kind: loop
    call: step
    times: 2
    input:
      n: \${input.start}
      meta: SEED
output: \${repeat.output}
`;

test("loop template seed types the carried field", async () => {
  const loaded = await loadConformanceFlow(LOOP.replace("SEED", "{first: '${input.start}'}"));

  const contract = loaded.compiled.nodes.get("repeat")!.outputContract as ValueContract;
  expect(canonicalTypeLabel(contract.type)).toBe('{"meta":{"first":int},"n":int}');
  const result = await runFlow(loaded, { start: 3 });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 5n, meta: { first: 3n } });
});

/** Return a flow whose inline LOOP body returns its carried `state` record. */
const inlineLoop = (seed: string) => `id: inline_loop
name: inline_loop
input:
  start: !int
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      state: ${seed}
    nodes:
      bump:
        kind: code
        input:
          state: \${input.state}
        output: STATE_TYPE
        code: |
          return {"state": {"n": inputs["state"]["n"] + 1}}
    output: \${bump.output}
output: \${repeat.output}
`;

test("inline loop template seed generates its parameter", async () => {
  const text =
    "typedefs:\n  State:\n    n: int\n" +
    inlineLoop("{n: '${input.start}'}").replace("STATE_TYPE", "\n          state: State");

  const result = await runFlow(await loadConformanceFlow(text), { start: 1 });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ state: { n: 3n } });
});

test("inline loop template seed mismatch suggests a tag", async () => {
  const text =
    "typedefs:\n  State:\n    n: float\n" +
    inlineLoop("{n: '${input.start}'}").replace("STATE_TYPE", "\n          state: State");

  const error = await loadErrorOf(text);

  expect(error.diagnostic.message).toContain(
    "the body returns 'state' as {\"n\":float}, but the carried field is typed " +
      "{\"n\":int} from its seed; declare the seed's type: `state: !State ",
  );
});

// ---------- inline bodies ----------

/** Return a flow whose inline CALL body reads `payload` bound as `binding`. */
const inlineCall = (binding: string) => `id: inline_call
name: inline_call
input:
  text: !str
nodes:
  c:
    kind: call
    input:
      payload: ${binding}
    nodes:
      echo:
        kind: code
        input:
          x: \${input.payload}
        code: |
          return inputs["x"]
    output: \${echo.output}
output: \${c.output}
`;

test("inline call template arg generates its parameter", async () => {
  const loaded = await loadConformanceFlow(inlineCall("{text: '${input.text}', tags: ['a', 'b']}"));

  expect(onlyParamLabel(loaded)).toBe('{"tags":list(str),"text":str}');
  const result = await runFlow(loaded, { text: "hi" });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ text: "hi", tags: ["a", "b"] });
});

test("inline call template arg with a nested tag uses it", async () => {
  const loaded = await loadConformanceFlow(inlineCall("{text: '${input.text}', items: !list(str) []}"));

  expect(onlyParamLabel(loaded)).toBe('{"items":list(str),"text":str}');
});

test("nested typed binding in a flow output is named by its path", async () => {
  const text = codeFlow("${p.output}").replace("output: ${c.output}", "output:\n  r:\n    a: !int ${p.output}");

  const error = await loadErrorOf(text);

  expect(error.diagnostic.message).toBe("output 'r.a': declared int, but the value is str");
  expect(errorLine(error)).toBe(lineOf(text, "    a: !int ${p.output}"));
});

test("inline call template list joins optional items", async () => {
  const text = inlineCall("['${input.text}', '${input.note}']").replace(
    "  text: !str\n",
    "  text: !str\n  note: !Optional(str)\n",
  );

  expect(onlyParamLabel(await loadConformanceFlow(text))).toBe("list(Optional(str))");
});

// ---------- nested typed containers with no strings ----------

test.each([
  "{k: '${p.output}', inner: !Obj {n: !int 3}}",
  "{k: '${p.output}', xs: !list(int) [!int 3]}",
  "!list(int) [!int 3]",
])("nested typed container holding only bindings loads: %s", async (binding) => {
  await loadConformanceFlow(codeFlow(binding));
});

test("inline call nested typed container holding only bindings loads", async () => {
  const text = TYPEDEFS + inlineCall("{inner: !Obj {n: !int 3}}");

  const result = await runFlow(await loadConformanceFlow(text), { text: "hi" });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ inner: { n: 3n } });
});

// ---------- literal leaves ----------

const LITERAL_LOOP = `id: literal_loop
name: literal_loop
input:
  start: !int
defs:
  step:
    nodes:
      bump:
        kind: code
        output: int
        code: |
          return 7
    output:
      st: {phase: scan, n: '\${bump.output}'}
nodes:
  repeat:
    kind: loop
    call: step
    times: 2
    input:
      st: {phase: init, n: '\${input.start}'}
output: \${repeat.output}
`;

test("loop seed differing from the body only in a literal runs", async () => {
  const result = await runFlow(await loadConformanceFlow(LITERAL_LOOP), { start: 1 });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ st: { phase: "scan", n: 7n } });
});

// ---------- partial templates ----------

/** Drop `p`'s declared output type and make it return an int. */
const untypedP = (text: string) => text.replace("    output: str\n", "").replace('return "hi"', "return 5");

test("partial template under a tag is checked at runtime", async () => {
  const flow = await loadConformanceFlow(untypedP(codeFlow("!Rec {a: '${p.output}', n: 1}")));

  const result = await runFlow(flow, {});

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
});
